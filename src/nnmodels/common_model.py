"""
Common class for all the convolutional neural network model used for image segmentation.
"""
try:
    from .model import NNModel

    import random

    from keras.preprocessing.image import img_to_array, load_img, array_to_img
    import glob
    import numpy as np
    from PIL import Image
    from scipy.misc import imresize
    from imageio import imwrite
    from skimage.transform import resize
    from utils import preprocessing as prep
    from utils import postprocessing as postp
    from os.path import splitext, basename, join, exists
    from os.path import split as pathsplit
    import time
except ImportError as err:
    exit("{}: {}".format(__file__, err))


class CommonModel(NNModel):
    """
    Common methods and attributes for neural network models.
    """

    ELEMENTS = [
        "background",
        "building"
    ]

    COLORS = [
        [0, 0, 0],       # background
        [128, 64, 128],  # building
    ]

    ELEMENTS_LABELS = dict(zip(ELEMENTS, COLORS))

    def __init__(self, model_name, input_shape=(256, 256, 3)):
        """
        Initialization of the model.
        """
        super().__init__(model_name=model_name)

        # Number of classes to segment
        # 0 -> not a building
        # 1 -> a building
        self.n_classes = 2
        # Input data shape
        self.input_shape = input_shape
        # File extensions for data to predict
        self.FILE_EXTENSIONS = [
            "tif",
            "tiff",
            "png",
            "jpg",
            "jpeg"
        ]
        # Files to predict
        self.filenames = list()

    def load_files_to_predict(self, files):
        """
        Loads files to predict.
        """
        self.filenames = files

    def predict_output(self):
        """
        Predicts an output for a given list of files/data.
        """
        for filename in self.filenames:
            print(filename)
            # Open the desired picture
            im = Image.open(filename)
            # Get the image array
            img_to_predict = np.array(im)
            # Be careful -> each pixel value must be a float
            img_to_predict = img_to_predict.astype('float32')
            # Close the file pointer (if possible)
            im.close()
            # Store the real shape for later
            real_shape = img_to_predict.shape

            # At this time we can only use images of shape (m*500, n*500, 3)
            assert real_shape[0] % 500 == 0
            assert real_shape[1] % 500 == 0

            # Predict the segmentation for this picture (its array is stored in data)
            pred = np.zeros(real_shape[:2] + (1,), dtype=int)
            for i in range(int(real_shape[0] / 500)):
                for j in range(int(real_shape[1] / 500)):
                    print(i, j)
                    # Get a sub-array of the main array
                    sub_array = img_to_predict[i * 500:(i + 1) * 500:, j * 500:(j + 1) * 500:, :]
                    sub_img = array_to_img(sub_array).resize(self.input_shape[:2])
                    # Because array_to_img is modifying array values to [0,255] we have
                    # to divide each value by 255
                    sub_array = np.array(sub_img) / 255.

                    # Predict the segmentation for this sub-array
                    pred_array = self._model.predict(sub_array.reshape((1,) + sub_array.shape))
                    # Get each maximum argument of this predicted array and don't
                    # forget to delete the first dimension of the predicted array
                    pred_array = np.argmax(pred_array.reshape(pred_array.shape[1:]), axis=2).astype(int)
                    # Resize the predicted array
                    resized_pred_array = resize(pred_array, (500, 500, 1), preserve_range=True, mode="edge").astype(
                        int)
                    # Add this sub-array to the main array
                    pred[i * 500:(i + 1) * 500:, j * 500:(j + 1) * 500:, :] = resized_pred_array

            # Transform pred array values to int values
            pred = pred.astype(int)
            # Create a numpy array with the RoadsModel.COLORS list
            colors = np.array(self.COLORS)
            # For each predicted value, get its associated color
            img_array = colors[pred.reshape(pred.shape[:2])]
            # Create a mask to get rid of [0,0,0] couples (aka background)
            mask = img_array != [0, 0, 0]
            # For each pixel in the image to predict, replace it with
            # the associated pixel from the predicted array. Only pixels
            # which are different to [0,0,0] will added.
            img_to_predict[mask] = img_array[mask]

            # Resize this array to the real shape of the image
            img_to_predict = resize(img_to_predict, real_shape, preserve_range=True, mode="edge")
            # Create a new Image instance with the new_img_array array
            new_img = Image.fromarray(img_to_predict.astype('uint8'))
            # Finally, save this image
            new_img.save(basename(splitext(filename)[0]) + "_segmented_img_test.jpg")
            # Save the unsegmented image
            imwrite(basename(splitext(filename)[0]) + "_unsegmented_img_test.jpg", np.array(Image.open(filename)))

            # Hold on, close the pointers before leaving
            new_img.close()

            print("Done")

    def create_generator(self, folder, batch_size=2):
        x_dir = join(folder, "x")
        y_dir = join(folder, "y")

        assert exists(x_dir) is True
        assert exists(y_dir) is True

        # FIX: glob.glob is waaaaay faster than [f for f in listdir() if isfile(f)]
        xfiles = [f for f_ in [glob.glob(join(x_dir, '*.' + e)) for e in self.FILE_EXTENSIONS] for f in f_]
        yfiles = [f for f_ in [glob.glob(join(y_dir, '*.' + e)) for e in self.FILE_EXTENSIONS] for f in f_]

        assert len(xfiles) == len(yfiles)
        assert len(xfiles) % batch_size == 0

        # Copy
        x_files = xfiles.copy()
        y_files = yfiles.copy()
        while True:
            x, y = list(), list()
            for _ in range(batch_size):
                if not x_files:
                    x_files = xfiles.copy()
                    y_files = yfiles.copy()

                # Get a new index
                index = x_files.index(random.choice(x_files))

                # MUST be true (files must have the same name)
                assert pathsplit(x_files[index])[-1] == pathsplit(y_files[index])[-1]

                x_img = img_to_array(load_img(x_files[index]))
                y_img = img_to_array(load_img(y_files[index]))

                # Resize each image
                x_img = imresize(x_img, self.input_shape[:2])
                y_img = imresize(y_img, self.input_shape[:2])
                # Apply a transformation on these images
                # x_img, y_img = prep.transform_xy(x_img, y_img)
                # Change y shape : (m, n, 3) -> (m, n, 2) (2 is the class number)
                temp_y_img = np.zeros(self.input_shape[:2] + (self.n_classes,))
                temp_y_img[y_img[:, :, 0] == 0] = [1, 0]
                temp_y_img[y_img[:, :, 0] == 255] = [0, 1]
                y_img = temp_y_img

                assert y_img.shape[2] == self.n_classes

                # Convert to float
                x_img = x_img.astype('float32')
                y_img = y_img.astype('float32')
                # Divide by the maximum value of each pixel
                x_img /= 255
                # Append images to the lists
                x.append(x_img)
                y.append(y_img)

                # Delete these elements
                del(x_files[index])
                del(y_files[index])
            yield np.array(x), np.array(y)

    # Abstract methods
    def create_layers(self):
        """
        Creates each layer of the model.
        """
        raise NotImplementedError("Please implement this method.")

    def learn(self):
        """
        Compiles and fits a model, evaluation is optional.
        """
        raise NotImplementedError("Please implement this method.")


if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
