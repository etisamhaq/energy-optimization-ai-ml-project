"""
Energy Optimization ML - Main package.
"""

__version__ = "1.0.0"
__author__ = "Energy ML Team"
__email__ = "energy-ml@example.com"

# Import main components for easier access
from . import data
from . import features  
from . import models
from . import optimization
from . import api
from . import visualization
from . import utils

__all__ = [
    'data',
    'features',
    'models', 
    'optimization',
    'api',
    'visualization',
    'utils'
]