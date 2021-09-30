import io
import sys
import json
import tqdm
import datetime
import importlib
import contextlib

import numpy as np

import torch
import torch.utils.data as td

import torchvision as tv

from PIL import Image

from collections import OrderedDict

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional


@contextlib.contextmanager
def tqdm_stdout(orig_stdout: Optional[io.TextIOBase] = None):

    class DummyFile(object):
        file = None

        def __init__(self, file):
            self.file = file

        def write(self, x):
            if len(x.rstrip()) > 0:
                tqdm.tqdm.write(x, file=self.file)

        def flush(self):
            return getattr(self.file, 'flush', lambda: None)()

    orig_out_err = sys.stdout, sys.stderr

    try:
        if orig_stdout is None:
            sys.stdout, sys.stderr = map(DummyFile, orig_out_err)
            yield orig_out_err[0]
        else:
            yield orig_stdout
    except Exception as exc:
        raise exc
    finally:
        sys.stdout, sys.stderr = orig_out_err


def load_class(package_name: str, class_name: Optional[str] = None) -> Type:
    if class_name is None:
        package_name, class_name = package_name.rsplit('.', 1)

    importlib.invalidate_caches()

    package = importlib.import_module(package_name)
    cls = getattr(package, class_name)

    return cls


def arg_selector(arg_cmd: Optional[Any], arg_conf: Optional[Any], arg_const: Any) -> Any:
    if arg_cmd is not None:
        return arg_cmd
    else:
        if arg_conf is not None:
            return arg_conf
        else:
            return arg_const


def collate_fn(batch):
    batch_audio, batch_image, batch_text = zip(*batch)

    keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_audio, batch_image))]

    if not all(audio is None for audio in batch_audio):
        batch_audio = [batch_audio[idx] for idx in keep_ids]
        batch_audio = torch.stack(batch_audio)
    else:
        batch_audio = None

    if not all(image is None for image in batch_image):
        batch_image = [batch_image[idx] for idx in keep_ids]
        batch_image = torch.stack(batch_image)
    else:
        batch_image = None

    if not all(text is None for text in batch_text):
        batch_text = [batch_text[idx] for idx in keep_ids]
    else:
        batch_text = None

    return batch_audio, batch_image, batch_text


def get_data_loaders(Dataset: Type,
                     dataset_args: Dict[str, Any],
                     batch_train: int = 64,
                     batch_test: int = 1024,
                     workers_train: int = 0,
                     workers_test: int = 0,
                     transforms_train: Optional[Callable[
                         [Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]
                     ]] = None,
                     transforms_test: Optional[Callable[
                         [Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]
                     ]] = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    dl_shuffle = dataset_args.pop('dl_shuffle', True)

    dataset_mode_train = {dataset_args['training']['key']: dataset_args['training']['yes']}
    dataset_mode_test = {dataset_args['training']['key']: dataset_args['training']['no']}

    dataset_args_train = {**{k: v for k, v in dataset_args.items() if k != 'training'}, **dataset_mode_train}
    dataset_args_test = {**{k: v for k, v in dataset_args.items() if k != 'training'}, **dataset_mode_test}

    ds_train = Dataset(**{
        **dataset_args_train,
        **{'transform_audio': transforms_train},
        **{'transform_frames': tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(224, interpolation=Image.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])}
    })
    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_train,
        shuffle=dl_shuffle,
        num_workers=workers_train,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    ds_eval = Dataset(**{
        **dataset_args_test,
        **{'transform_audio': transforms_test},
        **{'transform_frames': tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(224, interpolation=Image.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])}
    })
    eval_loader = torch.utils.data.DataLoader(
        ds_eval,
        batch_size=batch_test,
        num_workers=workers_test,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, eval_loader


def build_summary_str(experiment_name: str,
                      model_short_name: str,
                      model_class: str,
                      model_args: Dict[str, Any],
                      optimizer_class: str,
                      optimizer_args: Dict[str, Any],
                      dataset_class: str,
                      dataset_args: Dict[str, Any],
                      transforms: List[Dict[str, Union[str, Dict[str, Any]]]],
                      epochs: int,
                      batch_train: int,
                      log_interval: int,
                      saved_models_path: str,
                      scheduler_class: Optional[str] = None,
                      scheduler_args: Optional[Dict[str, Any]] = None) -> str:

    setup_title = '{}-{}'.format(experiment_name, model_short_name)

    summary_window_text = '<h3>'
    summary_window_text += '<a style="cursor: pointer;" onclick="jQuery(\'#{}\').toggle()">'.format(setup_title)

    summary_window_text += setup_title

    summary_window_text += '</a>'
    summary_window_text += '</h3>'
    summary_window_text += '<div style="margin: 5px; padding: 5px; background-color: lightgray;">'
    summary_window_text += '<div id="{}" style="display: none;"><pre>'.format(setup_title)

    summary = OrderedDict({
        'Date started': datetime.datetime.now().strftime('%Y-%m-%d @ %H:%M:%S'),
        'Model': OrderedDict({model_class: model_args}),
        'Setup': OrderedDict({
            'epochs': epochs,
            'batch': batch_train,
            'log_interval': log_interval,
            'saved_models_path': saved_models_path
        }),
        'Optimizer': OrderedDict({optimizer_class: optimizer_args}),
        'Dataset': OrderedDict({dataset_class: dataset_args}),
        'Transforms': OrderedDict({
            'Training': OrderedDict({tr['class']: tr['args'] for tr in transforms if tr['train']}),
            'Validation': OrderedDict({tr['class']: tr['args'] for tr in transforms if tr['test']})
        })
    })
    if scheduler_class is not None:
        summary['Scheduler'] = {scheduler_class: scheduler_args}
    summary_window_text += '{}'.format(
        json.dumps(summary, indent=2)
    )

    summary_window_text += '</pre></div>'
    summary_window_text += '</div>'

    return summary_window_text
