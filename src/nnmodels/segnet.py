try:
    from . import CommonModel

    from PIL import Image

    # Importing the required Keras modules containing models, layers, optimizers, losses, etc
    from keras.models import Model
    from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Concatenate, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.preprocessing.image import img_to_array, load_img, array_to_img
    from keras.optimizers import Adam, RMSprop
    from keras.losses import categorical_crossentropy, binary_crossentropy
    from keras.metrics import categorical_accuracy, binary_accuracy

    from os import listdir
    from os.path import isfile, exists, join, realpath, splitext, basename
    from os.path import split as pathsplit
    from random import randint

    import tensorflow as tf

    import glob

    import matplotlib.pyplot as plt

    import numpy as np

    import numpy as np
    from utils import preprocessing as prep
    from utils.losses import dice_coef_loss
    from .layers.pooling import MaxPoolingWithArgmax2D, MaxUnpooling2D
except ImportError as err:
    exit(err)


class SegNet(CommonModel):
    """
    SegNet model for image segmentation.

    Sources:
        - buildings dataset -> https://project.inria.fr/aerialimagelabeling/
    """

    def __init__(self):
        """
        Initialization of the model.
        """
        super().__init__(model_name=self.__class__.__name__.lower(), input_shape=(256, 256, 3))

    def create_layers(self):
        """
        Creates each layer of the model.
        """
        base_dir = join(realpath(__file__).split("src")[0], "data/aerial_buildings")
        train_dir = join(base_dir, "training")
        val_dir = join(base_dir, "validation")

        assert exists(train_dir) is True
        assert exists(val_dir) is True

        # Create a generator for each step
        train_generator = self.create_generator(train_dir, 8)  # 12600 images
        val_generator = self.create_generator(val_dir, 8)  # 5400 images

        # Datas
        self.datas = {"train_generator": train_generator, "val_generator": val_generator}

        # Inputs
        inputs = Input(self.input_shape)

        # Encoder
        conv_1 = Conv2D(8, (3, 3), padding="same")(inputs)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation(tf.nn.relu)(conv_1)
        conv_2 = Conv2D(8, (3, 3), padding="same")(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation(tf.nn.relu)(conv_2)

        pool_1, mask_1 = MaxPoolingWithArgmax2D((2, 2))(conv_2)

        conv_3 = Conv2D(16, (3, 3), padding="same")(pool_1)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation(tf.nn.relu)(conv_3)
        conv_4 = Conv2D(16, (3, 3), padding="same")(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation(tf.nn.relu)(conv_4)

        pool_2, mask_2 = MaxPoolingWithArgmax2D((2, 2))(conv_4)

        conv_5 = Conv2D(32, (3, 3), padding="same")(pool_2)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation(tf.nn.relu)(conv_5)
        conv_6 = Conv2D(32, (3, 3), padding="same")(conv_5)
        conv_6 = BatchNormalization()(conv_6)
        conv_6 = Activation(tf.nn.relu)(conv_6)
        conv_7 = Conv2D(32, (3, 3), padding="same")(conv_6)
        conv_7 = BatchNormalization()(conv_7)
        conv_7 = Activation(tf.nn.relu)(conv_7)

        pool_3, mask_3 = MaxPoolingWithArgmax2D((2, 2))(conv_7)

        conv_8 = Conv2D(64, (3, 3), padding="same")(pool_3)
        conv_8 = BatchNormalization()(conv_8)
        conv_8 = Activation(tf.nn.relu)(conv_8)
        conv_9 = Conv2D(64, (3, 3), padding="same")(conv_8)
        conv_9 = BatchNormalization()(conv_9)
        conv_9 = Activation(tf.nn.relu)(conv_9)
        conv_10 = Conv2D(64, (3, 3), padding="same")(conv_9)
        conv_10 = BatchNormalization()(conv_10)
        conv_10 = Activation(tf.nn.relu)(conv_10)

        pool_4, mask_4 = MaxPoolingWithArgmax2D((2, 2))(conv_10)

        conv_11 = Conv2D(128, (3, 3), padding="same")(pool_4)
        conv_11 = BatchNormalization()(conv_11)
        conv_11 = Activation(tf.nn.relu)(conv_11)
        conv_12 = Conv2D(128, (3, 3), padding="same")(conv_11)
        conv_12 = BatchNormalization()(conv_12)
        conv_12 = Activation(tf.nn.relu)(conv_12)
        conv_13 = Conv2D(128, (3, 3), padding="same")(conv_12)
        conv_13 = BatchNormalization()(conv_13)
        conv_13 = Activation(tf.nn.relu)(conv_13)

        pool_5, mask_5 = MaxPoolingWithArgmax2D((2, 2))(conv_13)
        print("Build encoder done..")

        # Decoder
        unpool_1 = MaxUnpooling2D((2, 2))([pool_5, mask_5])

        conv_14 = Conv2D(128, (3, 3), padding="same")(unpool_1)
        conv_14 = BatchNormalization()(conv_14)
        conv_14 = Activation(tf.nn.relu)(conv_14)
        conv_15 = Conv2D(128, (3, 3), padding="same")(conv_14)
        conv_15 = BatchNormalization()(conv_15)
        conv_15 = Activation(tf.nn.relu)(conv_15)
        conv_16 = Conv2D(64, (3, 3), padding="same")(conv_15)
        conv_16 = BatchNormalization()(conv_16)
        conv_16 = Activation(tf.nn.relu)(conv_16)

        unpool_2 = MaxUnpooling2D((2, 2))([conv_16, mask_4])

        conv_17 = Conv2D(64, (3, 3), padding="same")(unpool_2)
        conv_17 = BatchNormalization()(conv_17)
        conv_17 = Activation(tf.nn.relu)(conv_17)
        conv_18 = Conv2D(64, (3, 3), padding="same")(conv_17)
        conv_18 = BatchNormalization()(conv_18)
        conv_18 = Activation(tf.nn.relu)(conv_18)
        conv_19 = Conv2D(32, (3, 3), padding="same")(conv_18)
        conv_19 = BatchNormalization()(conv_19)
        conv_19 = Activation(tf.nn.relu)(conv_19)

        unpool_3 = MaxUnpooling2D((2, 2))([conv_19, mask_3])

        conv_20 = Conv2D(32, (3, 3), padding="same")(unpool_3)
        conv_20 = BatchNormalization()(conv_20)
        conv_20 = Activation(tf.nn.relu)(conv_20)
        conv_21 = Conv2D(32, (3, 3), padding="same")(conv_20)
        conv_21 = BatchNormalization()(conv_21)
        conv_21 = Activation(tf.nn.relu)(conv_21)
        conv_22 = Conv2D(16, (3, 3), padding="same")(conv_21)
        conv_22 = BatchNormalization()(conv_22)
        conv_22 = Activation(tf.nn.relu)(conv_22)

        unpool_4 = MaxUnpooling2D((2, 2))([conv_22, mask_2])

        conv_23 = Conv2D(16, (3, 3), padding="same")(unpool_4)
        conv_23 = BatchNormalization()(conv_23)
        conv_23 = Activation(tf.nn.relu)(conv_23)
        conv_24 = Conv2D(8, (3, 3), padding="same")(conv_23)
        conv_24 = BatchNormalization()(conv_24)
        conv_24 = Activation(tf.nn.relu)(conv_24)

        unpool_5 = MaxUnpooling2D((2, 2))([conv_24, mask_1])

        conv_25 = Conv2D(8, (3, 3), padding="same")(unpool_5)
        conv_25 = BatchNormalization()(conv_25)
        conv_25 = Activation(tf.nn.relu)(conv_25)

        conv_26 = Conv2D(self.n_classes, (1, 1), padding="same")(conv_25)
        conv_26 = BatchNormalization()(conv_26)
        conv_26 = Activation(tf.nn.softmax)(conv_26)

        # Set a new model with the inputs and the outputs (tenth convolution)
        self.set_model(Model(inputs=inputs, outputs=conv_26))

        # Get a summary of the previously create model
        self.get_model().summary()

    def learn(self):
        """
        Compiles and fits a model, evaluation is optional.
        """
        # Starting the training
        self._training = True
        # Create a new file to log the outputs
        self.create_logfile()

        # Number of epochs
        epochs = 10
        # Learning rate
        learning_rate = 1e-4
        # Compiling the model with an optimizer and a loss function
        self._model.compile(optimizer=Adam(lr=learning_rate),
                            loss=binary_crossentropy,
                            metrics=[binary_accuracy])

        # Fitting the model by using our train and validation data
        # It returns the history that can be plot in the future
        if "train_generator" in self.datas and "val_generator" in self.datas:
            # Fit including validation data
            self._history = self._model.fit_generator(
                self.datas["train_generator"],
                steps_per_epoch=1575,
                epochs=epochs,
                validation_data=self.datas["val_generator"],
                validation_steps=675)
        elif "train_generator" in self.datas:
            # Fit without validation datas
            self._history = self._model.fit_generator(
                self.datas["train_generator"],
                steps_per_epoch=1575,
                epochs=epochs)
        else:
            raise NotImplementedError("Unknown data")

        if "test_generator" in self.datas:
            # Evaluation of the model
            test_loss, acc_test = self._model.evaluate_generator(self.datas["test_generator"], steps=250, verbose=1)
            print("Loss / test: " + str(test_loss) + " and accuracy: " + str(acc_test))

        # Training is over
        self._training = False


if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
