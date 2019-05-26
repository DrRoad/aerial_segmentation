try:
    from keras import backend as K
    import numpy as np
except ImportError as err:
    exit(err)


def dice_coef(y_true, y_pred, smooth=1):
    """
    From: https://github.com/keras-team/keras/issues/3611 and
    https://github.com/keras-team/keras/issues/3611#issuecomment-243108708
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth)/(union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    """
    From: https://github.com/keras-team/keras/issues/3611 and
    https://github.com/keras-team/keras/issues/3611#issuecomment-243108708
    """
    return 1 - dice_coef(y_true, y_pred)


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    From: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08

    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # Skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean(numerator / (denominator + epsilon))  # Average over classes and batch
