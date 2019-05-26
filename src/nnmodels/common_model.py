"""
Common class for all the convolutional neural network model used for image segmentation.
"""
try:
    from .model import NNModel

    from keras.preprocessing.image import img_to_array, load_img, array_to_img

    import numpy as np
    from PIL import Image
    from imageio import imwrite
    from skimage.transform import resize
    from utils import preprocessing as pp
    from os.path import splitext, basename
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
            new_img.save(basename(splitext(filename)[0]) + "_segmented_img.jpg")
            # Save the unsegmented image
            imwrite(basename(splitext(filename)[0]) + "_unsegmented_img.jpg", np.array(Image.open(filename)))

            # Hold on, close the pointers before leaving
            new_img.close()

            print("Done")

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
