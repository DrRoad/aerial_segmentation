try:
    from keras import backend as K
    import numpy as np
except ImportError as err:
    exit(err)


def dice_coef(y_true, y_pred, loss_type='sorensen', axis=None, smooth=1e-5):
    """
    Dice coefficient calculation.

    From: https://github.com/keras-team/keras/issues/3611 and
    https://github.com/keras-team/keras/issues/3611#issuecomment-243108708

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
    """
    if axis is None:
        axis = [1, 2, 3]

    if loss_type == 'jaccard':
        t = K.sum(K.square(y_true), axis=axis)
        p = K.sum(K.square(y_pred), axis=axis)
    elif loss_type == 'sorensen':
        t = K.sum(y_true, axis=axis)
        p = K.sum(y_pred, axis=axis)
    else:
        raise Exception("Unknown loss_type: {}".format(loss_type))

    intersection = K.sum(y_true * y_pred, axis=axis)
    union = t + p
    return K.mean((2. * intersection + smooth)/(union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    """
    Dice coefficient loss function.

    From: https://github.com/keras-team/keras/issues/3611 and
    https://github.com/keras-team/keras/issues/3611#issuecomment-243108708

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
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


def dice_coef_test(y_true, y_pred, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, num_labels=2):
    dice = 0
    for index in range(num_labels):
        dice += dice_coef_test(y_true[:, :, :, index], y_pred[:, :, :, index])
    return 1 - dice


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (1, 2, 3)) + beta * K.sum(p1 * g0, (1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T
