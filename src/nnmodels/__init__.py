"""
TODO
"""
from .model        import NNModel
from .common_model import CommonModel
from .segnet       import SegNet
from .vnet         import VNet
from .vegnet       import VegNet


# This list contains all of the available neural network models
available_models = {
    SegNet.__name__.lower() : SegNet(),
    VNet.__name__.lower()   : VNet(),
    VegNet.__name__.lower() : VegNet
}
