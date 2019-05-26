try:
    import numpy as np
    from os import listdir
    from os.path import exists, join, isfile, splitext, basename
    from PIL import Image
    from skimage.transform import resize
    from imageio import imwrite
    from keras.preprocessing.image import array_to_img
    from keras.utils import to_categorical
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
        >>> arr = np.array([3, 2, 4, 0, 4, 0, 2, 1, 0, 1])
        >>> old = np.array([0, 1, 3])
        >>> new = np.array([29, 42, 13])
        >>> a   = change_values_in_array(a, old, new)
        >>> print(a)
        [13  2  4 29  4 29  2 42 29 42]

    Adapted from:
        https://stackoverflow.com/questions/29407945/find-and-replace-multiple-values-in-python (Ashwini_Chaudhary solution)
    """
    try:
        arr = np.arange(a.max() + 1, dtype=val_new.dtype)
        arr[val_old] = val_new
        return arr[a]
    except IndexError as e:
        val = int(str(e).split(" is out")[0].split("index ")[1])
        index = np.where(val_old == val)[0][0]
        return change_values_in_array(a, val_old[val_old != val], np.delete(val_new, index))


def test_change_values_in_array():
    try:
        import time
        from keras.utils import to_categorical
    except ImportError as e:
        exit("{}: {}".format(__file__, e))

    a = np.random.randint(65, size=(2000, 2000))
    val_old = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                        10, 11, 12,     14, 15, 16, 17, 18, 19,
                        20, 21, 22,         25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47,
                            51, 52, 53, 54,     56, 57, 58, 59,
                        60,     62, 63, 64, 65])
    val_new = np.array([0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,      0,  0,  0,  0,  0,  0,
                        0,  0,  0,          0,  0,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0,  0,
                            0,  0,  0,  0,      0,  0,  0,  0,
                        0,      0,  0,  0,  0])

    start = time.time()
    im = change_values_in_array(a, val_old, val_new)
    end = time.time()
    temp_arr = np.zeros(im.shape[:2] + (8,))
    # Then, for each value, create a binary list of length 8
    # (well known as "one hot encoding")
    print(temp_arr.shape)
    print(im.shape)
    temp_arr[im[:, :] == 0]  = [1, 0, 0, 0, 0, 0, 0, 0]
    temp_arr[im[:, :] == 13] = [0, 1, 0, 0, 0, 0, 0, 0]
    temp_arr[im[:, :] == 48] = [0, 0, 1, 0, 0, 0, 0, 0]
    temp_arr[im[:, :] == 49] = [0, 0, 0, 1, 0, 0, 0, 0]
    temp_arr[im[:, :] == 50] = [0, 0, 0, 0, 1, 0, 0, 0]
    temp_arr[im[:, :] == 55] = [0, 0, 0, 0, 0, 1, 0, 0]
    temp_arr[im[:, :] == 61] = [0, 0, 0, 0, 0, 0, 1, 0]
    temp_arr[im[:, :] == 23] = [0, 0, 0, 0, 0, 0, 0, 1]
    temp_arr[im[:, :] == 24] = [0, 0, 0, 0, 0, 0, 0, 1]
    print("temps:", end - start)
    print(im)
    print(im.mean())
    unique, counts = np.unique(im, return_counts=True)
    d = dict(zip(unique, counts))
    print("0:", d[0], ", 1:", d[13], ", 2:", d[23]+d[24], ", 3:", d[48], ", 4:", d[49]+d[50], ", 5:", d[55]+d[61])

    unique, counts = np.unique(a, return_counts=True)
    d = dict(zip(unique, counts))
    print("0:", d[0], ", 1:", d[13], ", 2:", d[23]+d[24], ", 3:", d[48], ", 4:", d[49]+d[50], ", 5:", d[55]+d[61])

    old = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                    60, 61, 62, 63, 64, 65])
    new = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 2, 2, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 3, 4,
                    4, 0, 0, 0, 0, 5, 0, 0, 0, 0,
                    0, 5, 0, 0, 0, 0])
    start = time.time()
    y_img = change_values_in_array(a, old, new)
    end = time.time()
    to_categorical(y_img, num_classes=6, dtype=int)
    print("temps:", end - start)
    print(y_img)
    print(y_img.mean())
    unique, counts = np.unique(y_img, return_counts=True)
    d = dict(zip(unique, counts))
    print("0:", d[0], ", 1:", d[1], ", 2:", d[2], ", 3:", d[3], ", 4:", d[4], ", 5:", d[5])


def crop_and_resize(folder, result_dir):
    """
    Crop and resize a list of images. 
    """
    assert exists(folder) is True
    assert exists(result_dir) is True

    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    for file in files:
        print(file)
        # Get the filename and its extension
        filename, ext = splitext(basename(file))
        # Open the file as an PIL.Image instance
        img = Image.open(file)
        # Convert this Image to a numpy array
        im = np.array(img)
        # Get a new shape
        shape = transform_shape(im.shape)
        # Then resize this image with the new shape
        im = resize(im, shape)
        # For each subarray, create a new image
        for i in range(int(im.shape[0] / 1000)):
            for j in range(int(im.shape[1] / 1000)):
                if exists(join(result_dir, filename + "_{}{}".format(i, j) + ext)):
                    continue
                # Crop the image in a sub-array
                im_crop = im[i * 1000:(i + 1) * 1000, j * 1000:(j + 1) * 1000]
                # Resize it to reduce the shape
                im_crop_resized = resize(im_crop, (336, 336) + im_crop.shape[2:])
                # Finally, save it as a new image
                imwrite(join(result_dir, filename + "_{}{}".format(i, j) + ext), im_crop_resized)


def test_crop_and_resize():
    try:
        from os.path import realpath
        from os import makedirs
    except ImportError as e:
        exit("{}: {}".format(__file__, e))

    # Base directory
    base_dir = join(realpath(__file__).split("src")[0], "datas/mapillary_datasets")
    # Training dataset
    train_dir = join(base_dir, "training")
    train_images_dir = join(train_dir, "images")
    train_labels_dir = join(train_dir, "labels")
    resized_img_dir = join(train_dir, "x")
    resized_lbl_dir = join(train_dir, "y")

    assert exists(train_images_dir) is True
    assert exists(train_labels_dir) is True

    makedirs(resized_img_dir, exist_ok=True)
    makedirs(resized_lbl_dir, exist_ok=True)

    crop_and_resize(train_images_dir, resized_img_dir)
    crop_and_resize(train_labels_dir, resized_lbl_dir)

    # Validation dataset
    val_dir = join(base_dir, "validation")
    val_images_dir = join(val_dir, "images")
    val_labels_dir = join(val_dir, "labels")
    resized_img_dir = join(val_dir, "x")
    resized_lbl_dir = join(val_dir, "y")

    assert exists(val_images_dir) is True
    assert exists(val_labels_dir) is True

    makedirs(resized_img_dir, exist_ok=True)
    makedirs(resized_lbl_dir, exist_ok=True)

    crop_and_resize(val_images_dir, resized_img_dir)
    crop_and_resize(val_labels_dir, resized_lbl_dir)


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
    if len(shape) <= 1 or len(shape) > 3:
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
            sup = str(int(sup) + 1)
        new_shape += (int(sup + "000"),)

    # Don't forget the last element (only if it exists)
    if len(shape) == 3:
        new_shape += (shape[2],)

    return new_shape


def test_transform_shape():
    try:
        transform_shape((5120,))
    except ValueError:
        assert True
    else:
        assert False
    assert transform_shape((4850, 650)) == (5000, 1000)
    assert transform_shape((501, 321)) == (1000, 1000)
    assert transform_shape((499, 2432)) == (1000, 2000)
    assert transform_shape((11680, 6825)) == (12000, 7000)
    assert transform_shape((736825, 8025, 3)) == (737000, 8000, 3)


def test_show_label():
    _dir = "C:/Users/e_sgouge/Documents/Etienne/Python/roads_segmentation/datas/mapillary_datasets/training/y"
    file = join(_dir, "__M2DBwhxBjZgQXkk5kwjQ_12.png")

    img = np.array(Image.open(file)).astype(int)

    old = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47,
                    51, 52, 53, 54, 56, 57, 58, 59,
                    60, 62, 63, 64, 65])
    new = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0])
    img = change_values_in_array(img, old, new)

    # Convert to a binary matrix (shape might be equal to (m, n, 7))
    # FIX: because to_categorical is not useful in this case
    # we have to do this by hand
    # First of all, create an empy array
    temp_arr = np.zeros(img.shape[:2] + (1,), dtype=int)
    # Then, for each value, create a binary list of length 8
    # (well known as "one hot encoding")
    temp_arr[img[:, :] == 0] = 0
    temp_arr[img[:, :] == 13] = 1
    temp_arr[img[:, :] == 48] = 2
    temp_arr[img[:, :] == 49] = 3
    temp_arr[img[:, :] == 50] = 4
    temp_arr[img[:, :] == 55] = 5
    temp_arr[img[:, :] == 61] = 6
    temp_arr[img[:, :] == 23] = 7
    temp_arr[img[:, :] == 24] = 7

    img = temp_arr

    unique, counts = np.unique(img, return_counts=True)
    print(dict(zip(unique, counts)))

    COLORS = [
        [0, 0, 0],  # background
        [128, 64, 128],  # road
        [250, 170, 30],  # traffic light
        [192, 192, 192],  # traffic sign (back)
        [220, 220, 0],  # traffic sign (front)
        [0, 0, 142],  # car
        [0, 0, 70],  # truck
        [200, 128, 128]  # Lane marking
    ]
    colors = np.array(COLORS)
    # For each predicted value, get its associated color
    img = colors[img.reshape(img.shape[:2])]

    new_img = array_to_img(img)
    new_img.save("test_label_2.jpg")


if __name__ == "__main__":
    test_difference_between_two_files()
