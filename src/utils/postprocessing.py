"""
TODO
"""
try:
    import numpy as np
    # Dense CRF and useful functions
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
except ImportError as e:
    exit(e)


def apply_crf_on_output(img, out, n_classes):
    """
    Given an image and its segmentation, applies a Conditional Random
    Field (CRF) on the segmented image.

    :param img: default image
    :param out: segmented image
    :param n_classes: number of classes in the segmentation
    :return: `out` with a CRF applied on it
    """
    # Remove single-dimensional entries
    out = out.squeeze()
    # Put the last dimension in the first place (otherwise we can't use unary_from_softmax)
    out = out.transpose((2, 0, 1))
    # Get unary potentials (neg log probability)
    unary = unary_from_softmax(out)

    # The inputs should be C-continuous -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], n_classes)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    # It creates the color-independent features and then adds them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats,
                        compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(15, 15),
                                      schan=(20, 20, 20),
                                      img=img,
                                      chdim=2)
    d.addPairwiseEnergy(feats,
                        compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    q = d.inference(10)
    res = np.argmax(q, axis=0).reshape((img.shape[0], img.shape[1]))

    return res
