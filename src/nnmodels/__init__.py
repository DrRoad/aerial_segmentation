"""
TODO
"""
from .model        import NNModel
from .common_model import CommonModel
from .segnet       import SegNet


# This list contains all of the available neural network models
available_models = {
    SegNet.__name__.lower() : SegNet()
}
