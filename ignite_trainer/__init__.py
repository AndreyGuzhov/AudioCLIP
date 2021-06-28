import os as _os
import sys as _sys

from ignite_trainer.version import __version__
from ._trainer import main, run
from ._utils import load_class
from ._interfaces import AbstractNet, AbstractTransform

__all__ = [
    '__version__',
    'main', 'run',
    'load_class',
    'AbstractNet', 'AbstractTransform'
]

_sys.path.extend([_os.getcwd()])
