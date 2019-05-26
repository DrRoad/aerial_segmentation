try:
    from os import listdir
    from os.path import exists, join, isfile, splitext, basename
    import numpy as np
    from PIL import Image
    from skimage.transform import resize, rotate
    from skimage.util import random_noise
    from random import uniform, choice, randint
    from keras.preprocessing.image import array_to_img
    from imageio import imwrite
except ImportError as err:
    exit("{}: {}".format(__file__, err))


def change_values_in_array(a, val_old, val_new):
    """
    Given a numpy array, it changes each specific value in another.
    This function is recursive.

    Attributes:
        - a      : the numpy array to work on
        - val_old: old values
        - val_new: new values

    Returns:
        A numpy array of the same shape.

    Example:
        >>> import numpy as np
        >>> arr  = np.array([3, 2, 4, 0, 4, 0, 2, 1, 0, 1])
        >>> old  = np.array([0, 1, 3])
        >>> new  = np.array([29, 42, 13])
        >>> _arr = change_values_in_array(a, old, new)
        >>> print(_arr)
        [13  2  4 29  4 29  2 42 29 42]

    Adapted from: https://stackoverflow.com/questions/29407945/find-and-replace-multiple-values-in-python (Ashwini_Chaudhary solution)
    """
    try:
        arr = np.arange(a.max()+1, dtype=val_new.dtype)
        arr[val_old] = val_new
        return arr[a]
    except IndexError as e:
        val = int(str(e).split(" is out")[0].split("index ")[1])
        index = np.where(val_old == val)[0][0]
        return change_values_in_array(a, val_old[val_old != val], np.delete(val_new, index))


def transform_shape(shape):
    """
    Transforms a shape to resize an image after that. Shape size must 
    be equal to 2 or 3, otherwise it raises a ValueError.

    Returns:
        A tuple of the same size.

    Raises:
        A ValueError if the shape size is not equal to 2 or 3.

    Example:
        >>> shape = (4850, 650, 3)
        >>> new_shape = transform_shape(shape)
        >>> print(new_shape)
        (5000, 1000, 3)
    """
    if len(shape) < 2 or len(shape) > 3:
        raise ValueError("ERROR: Shape size must be in [2;3]")
    # Create a tuple to store the new shape
    new_shape = tuple()
    for value in shape[:2]:
        # Convert this value to a string
        val = str(value)
        # Get the first two values and store the rest in another 
        # variable
        sup, inf = val[:-3] if val[:-3] != '' else "1", val[-3:]
        if int(inf) > 500:
            sup = str(int(sup)+1)
        new_shape += (int(sup + "000"),)

    # Don't forget the last element (only if it exists)
    if len(shape) == 3:
        new_shape += (shape[2],)

    return new_shape


def crop_and_resize_images(folder, result_dir, cropping_size=1000, new_size=(336, 336)):
    """
    Crops and resizes a list of images. 
    """
    assert exists(folder)     is True
    assert exists(result_dir) is True

    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    for f in files:
        print(f)
        # Get the filename and its extension
        fname, ext = splitext(basename(f))
        # Open the file as an PIL.Image instance
        _img = Image.open(f)
        # Convert this Image to a numpy array
        im = np.array(_img)
        # Get a new shape
        shape = transform_shape(im.shape)
        # Then resize this image with the new shape
        im = resize(im, shape, preserve_range=True)
        # For each subarray, create a new image
        for i in range(int(im.shape[0]/cropping_size)):
            for j in range(int(im.shape[1]/cropping_size)):
                if exists(join(result_dir, fname + "_{}{}".format(i, j) + ext)):
                    continue
                # Crop the image in a sub-array
                im_crop = im[i*cropping_size:(i+1)*cropping_size, j*cropping_size:(j+1)*cropping_size]
                # Resize it to reduce the shape
                im_crop_resized = resize(im_crop, new_size + im_crop.shape[2:], preserve_range=True)
                # Finally, save it as a new image
                imwrite(join(result_dir, fname + "_{}{}".format(i, j) + ext), im_crop_resized)
        # Close the pointer
        img.close()


def resize_images(folder, result_dir, new_size=(768, 1024)):
    """
    Resizes a list of images.
    """
    assert exists(folder)     is True
    assert exists(result_dir) is True

    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    for f in files:
        print(f)
        # Get the filename and its extension
        fname, ext = splitext(basename(f))
        # Has this image ever been moved ?
        if exists(join(result_dir, fname + ext)):
            continue
        # Open the file as an PIL.Image instance
        im = Image.open(f)
        # Convert this Image to a numpy array
        im = np.array(im)
        # Resize it to reduce the shape
        im_resized = resize(im, new_size + im.shape[2:], preserve_range=True)
        # Finally, save it as a new image
        imwrite(join(result_dir, fname + ext), im_resized)
        # Close the pointer
        im.close()


def transfrom_xy(_x, _y):
    """
    Applies random transformations on x and y.
    """
    functions = list(DATA_AUGMENTATION_FUNCTION)
    # Flip (or not) both images
    flip_or_not = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[1]])]
    _x, _y = flip_or_not(_x, _y)
    # Rotate (or not) both images
    rotate_or_not = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[2]])]
    _x, _y = rotate_or_not(_x, _y)
    # Add noise (or not) to the first image
    # addNoiseOrNot = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[3]])]
    # x, y = addNoiseOrNot(x, y)
    # Zoom (or not) on both images
    zoom_or_not = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[4]])]
    _x, _y = zoom_or_not(_x, _y)

    return _x, _y


def _return_xy(_x, _y):
    """
    Returns x and y.
    """
    return _x, _y


def _flip_xy(_x, _y):
    """
    Flips x and y.
    """
    return np.fliplr(_x), np.fliplr(_y)


def _rotation_xy(_x, _y, rot=10, random_rot=True):
    """
    Rotation to x and y.
    If 'random_rot' is set to True it does a random rotation between [rot;rot].
    """
    if random_rot:
        # Pick a random degree of rotation between rot% on the left and rot% on the right
        rand_deg = uniform(-rot, rot)
    else:
        rand_deg = rot
    # IMPORTANT: set parameter "preserve_range" to True to have values in [0;255] instead of [0;1].
    _x, _y = rotate(_x, rand_deg, preserve_range=True), rotate(_y, rand_deg, preserve_range=True)
    return _x, _y


def _random_noise_xy(_x, _y):
    """
    Adds random noise to x but nothing to y.
    """
    dtype = _x.dtype
    nx = random_noise(_x) * 255.0
    if np.mean(nx) > np.mean(_x)+10:
        # Return x and y because random_noise returned a white image.
        return _x, _y
    return nx.astype(dtype), _y


def _zoom_xy(_x, _y, zoom=None):
    """
    Zooms on x and y. The same zoom is applied on both images.
    If no value is applied to 'zoom' a random value is set.
    """
    if zoom is None:
        zoom = randint(0, int(min(_x.shape[:2] + _y.shape[:2]) / 4))
    x_zoomed = _x[zoom:_x.shape[0] - zoom, zoom:_x.shape[1] - zoom, :]
    y_zoomed = _y[zoom:_y.shape[0] - zoom, zoom:_y.shape[1] - zoom, :]
    _x = np.array(array_to_img(x_zoomed).resize(_x.shape[:2][::-1]))
    _y = np.array(array_to_img(y_zoomed).resize(_y.shape[:2][::-1]))
    return _x, _y


DATA_AUGMENTATION_FUNCTION = {
    # No transformation
    'returnXY'      : _return_xy,
    # Transformations
    'flipXY'        : _flip_xy,
    'rotationXY'    : _rotation_xy,
    'randomNoiseXY' : _random_noise_xy,
    'zoomXY'        : _zoom_xy
}


if __name__ == "__main__":
    # Imports for testing
    try:
        from PIL import Image
        from os.path import exists, join
        from os import mkdir
        from scipy.misc import imsave
    except ImportError as err:
        exit("{}: {}".format(__file__, err))

    # Main directory of the images
    dir_path   = "C:/Users/e_sgouge/Documents/Etienne/Python/analyze_images"
    filename   = "route_2.jpg"
    file       = join(dir_path, filename)
    output_dir = join(dir_path, "tests")

    # Check if file and directory exist
    assert exists(dir_path) is True
    assert exists(file)     is True

    if not exists(output_dir):
        mkdir(output_dir)

    # Open the image
    img = Image.open(file)
    # Transform it to a numpy array
    x, y = np.array(img), np.array(img)

    # FLIP
    x_flip, y_flip = _flip_xy(x, y)
    imsave(join(output_dir, "flip_x_" + filename), x_flip)
    imsave(join(output_dir, "flip_y_" + filename), y_flip)

    # ROTATE
    x_rot, y_rot = _randomRotationXY(x, y)
    imsave(join(output_dir, "rot_x_" + filename), x_rot)
    imsave(join(output_dir, "rot_y_" + filename), y_rot)

    # NOISE
    x_noise, _ = _random_noise_xy(x, y)
    imsave(join(output_dir, "noise_x_" + filename), x_noise)

    # ZOOM
    x_zoom, y_zoom = _randomZoomXY(x, y)
    imsave(join(output_dir, "zoom_x_" + filename), x_zoom)
    imsave(join(output_dir, "zoom_y_" + filename), y_zoom)
