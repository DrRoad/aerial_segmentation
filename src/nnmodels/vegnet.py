try:
    from . import CommonModel

    from PIL import Image

    # Importing the required Keras modules containing models, layers, optimizers, losses, etc
    from keras.models import Model
    from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Concatenate, Activation, Add
    from keras.layers import Conv2DTranspose
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

    import matplotlib.pyplot as plt

    import numpy as np

    import numpy as np
    from utils import preprocessing as prep
    from utils.losses import dice_coef_loss
    from .layers.pooling import MaxPoolingWithArgmax2D, MaxUnpooling2D
except ImportError as err:
    exit(err)


class VegNet(CommonModel):
    """
    Mix between V-Net and SegNet models for image segmentation.

    Sources:
        - buildings dataset -> https://project.inria.fr/aerialimagelabeling/
    """

    def __init__(self):
        """
        Initialization of the model.
        """
        super().__init__(model_name=self.__class__.__name__.lower(), input_shape=(352, 352, 3))

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

        # Data
        self.data = {"train_generator": train_generator, "val_generator": val_generator}

        # Inputs
        inputs = Input(self.input_shape)
        # ----- First Convolution - Down-convolution -----
        # 5x5 Convolution
        conv1 = Conv2D(8, (5, 5), padding='same', data_format='channels_last', name='conv1_1')(inputs)
        acti1 = Activation(tf.nn.relu, name='acti1')(conv1)
        # Down-convolution
        pool_1, mask_1 = MaxPoolingWithArgmax2D((2, 2))(acti1)

        # ----- Second Convolution - Down-convolution -----
        # 5x5 Convolution
        conv2_1 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv2_1')(pool_1)
        acti2 = Activation(tf.nn.relu, name='acti2_1')(conv2_1)
        # 5x5 Convolution
        conv2 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv2_2')(acti2)
        acti2 = Activation(tf.nn.relu, name='acti2_2')(conv2)
        # Add layer
        add2 = Add(name='add2_1')([conv2_1, acti2])
        # Down-convolution
        pool_2, mask_2 = MaxPoolingWithArgmax2D((2, 2))(add2)

        # ----- Third Convolution - Down-convolution -----
        # 5x5 Convolution
        conv3_1 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv3_1')(pool_2)
        acti3 = Activation(tf.nn.relu, name='acti3_1')(conv3_1)
        # 5x5 Convolution
        conv3 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv3_2')(acti3)
        acti3 = Activation(tf.nn.relu, name='acti3_2')(conv3)
        # 5x5 Convolution
        conv3 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv3_3')(acti3)
        acti3 = Activation(tf.nn.relu, name='acti3_3')(conv3)
        # Add layer
        add3 = Add(name='add3_1')([conv3_1, acti3])
        # Down-convolution
        pool_3, mask_3 = MaxPoolingWithArgmax2D((2, 2))(add3)

        # ----- Fourth Convolution - Down-convolution -----
        # 5x5 Convolution
        conv4_1 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv4_1')(pool_3)
        acti4 = Activation(tf.nn.relu, name='acti4_1')(conv4_1)
        # 5x5 Convolution
        conv4 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv4_2')(acti4)
        acti4 = Activation(tf.nn.relu, name='acti4_2')(conv4)
        # 5x5 Convolution
        conv4 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv4_3')(acti4)
        acti4 = Activation(tf.nn.relu, name='acti4_3')(conv4)
        # Add layer
        add4 = Add(name='add4_1')([conv4_1, acti4])
        # Down-convolution
        pool_4, mask_4 = MaxPoolingWithArgmax2D((2, 2))(add4)

        # ----- Fifth Convolution -----
        # 5x5 Convolution
        conv5_1 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv5_1')(pool_4)
        acti5 = Activation(tf.nn.relu, name='acti5_1')(conv5_1)
        # 5x5 Convolution
        conv5 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv5_2')(acti5)
        acti5 = Activation(tf.nn.relu, name='acti5_2')(conv5)
        # 5x5 Convolution
        conv5 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv5_3')(acti5)
        acti5 = Activation(tf.nn.relu, name='acti5_3')(conv5)
        # Add layer
        add5 = Add(name='add5_1')([conv5_1, acti5])
        # Up-convolution
        pool_5, mask_5 = MaxPoolingWithArgmax2D((2, 2))(add5)

        # ----- Sixth Convolution -----
        # MaxUnpooling
        unpool_6 = MaxUnpooling2D((2, 2))([pool_5, mask_5])
        # Concatenation
        conc6 = Concatenate(axis=3, name='conc6')([unpool_6, add5])
        # 5x5 Convolution
        conv6 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv6_1')(conc6)
        acti6 = Activation(tf.nn.relu, name='acti6_1')(conv6)
        # 5x5 Convolution
        conv6 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv6_2')(acti6)
        acti6 = Activation(tf.nn.relu, name='acti6_2')(conv6)
        # 5x5 Convolution
        conv6 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv6_3')(acti6)
        acti6 = Activation(tf.nn.relu, name='acti6_3')(conv6)
        # Add layer
        add6 = Add(name='add6_1')([unpool_6, acti6])

        # ----- Seventh Convolution -----
        # 5x5 Convolution
        conv7 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv7_1')(add6)
        acti7 = Activation(tf.nn.relu, name='acti7_1')(conv7)
        # Up-convolution
        unpool_7 = MaxUnpooling2D((2, 2))([acti7, mask_4])
        # Concatenation
        conc7 = Concatenate(axis=3, name='conc7')([unpool_7, add4])
        # 5x5 Convolution
        conv7 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv7_2')(conc7)
        acti7 = Activation(tf.nn.relu, name='acti7_2')(conv7)
        # 5x5 Convolution
        conv7 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv7_3')(acti7)
        acti7 = Activation(tf.nn.relu, name='acti7_3')(conv7)
        # Add layer
        add7 = Add(name='add7_1')([unpool_7, acti7])

        # ----- Eighth Convolution -----
        # 5x5 Convolution
        conv8 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv8_1')(add7)
        acti8 = Activation(tf.nn.relu, name='acti8_1')(conv8)
        # Up-convolution
        unpool_8 = MaxUnpooling2D((2, 2))([acti8, mask_3])
        # Concatenation
        conc8 = Concatenate(axis=3, name='conc8')([unpool_8, add3])
        # 5x5 Convolution
        conv8 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv8_2')(conc8)
        acti8 = Activation(tf.nn.relu, name='acti8_2')(conv8)
        # 5x5 Convolution
        conv8 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv8_3')(acti8)
        acti8 = Activation(tf.nn.relu, name='acti8_3')(conv8)
        # Add layer
        add8 = Add(name='add8_1')([unpool_8, acti8])

        # ----- Ninth Convolution -----
        # 5x5 Convolution
        conv9 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv9_1')(add8)
        acti9 = Activation(tf.nn.relu, name='acti9_1')(conv9)
        # Up-convolution
        unpool_9 = MaxUnpooling2D((2, 2))([acti9, mask_2])
        # Concatenation
        conc9 = Concatenate(axis=3, name='conc9')([unpool_9, add2])
        # 5x5 Convolution
        conv9 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv9_2')(conc9)
        acti9 = Activation(tf.nn.relu, name='acti9_2')(conv9)
        # Add layer
        add9 = Add(name='add9_1')([unpool_9, acti9])

        # ----- Tenth Convolution -----
        # 5x5 Convolution
        conv10 = Conv2D(8, (5, 5), padding='same', data_format='channels_last', name='conv10_1')(add9)
        acti10 = Activation(tf.nn.relu, name='acti10_1')(conv10)
        # Up-convolution
        unpool_10 = MaxUnpooling2D((2, 2))([acti10, mask_1])

        # ----- Eleventh Convolution -----
        conv11 = Conv2D(self.n_classes, (1, 1), padding='same', data_format='channels_last', name='conv11_1')(unpool_10)
        acti11 = Activation(tf.nn.softmax, name='acti11_1')(conv11)

        # Set a new model with the inputs and the outputs (tenth convolution)
        self.set_model(Model(inputs=inputs, outputs=acti11))

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
        learning_rate = 1e-3
        # Compiling the model with an optimizer and a loss function
        self._model.compile(optimizer=Adam(lr=learning_rate),
                            loss=binary_crossentropy,
                            metrics=[binary_accuracy])

        # Fitting the model by using our train and validation data
        # It returns the history that can be plot in the future
        if "train_generator" in self.data and "val_generator" in self.data:
            # Fit including validation datas
            self._history = self._model.fit_generator(
                self.data["train_generator"],
                steps_per_epoch=1575,
                epochs=epochs,
                validation_data=self.data["val_generator"],
                validation_steps=675)
        elif "train_generator" in self.data:
            # Fit without validation datas
            self._history = self._model.fit_generator(
                self.data["train_generator"],
                steps_per_epoch=1575,
                epochs=epochs)
        else:
            raise NotImplementedError("Unknown data")

        if "test_generator" in self.data:
            # Evaluation of the model
            test_loss, acc_test = self._model.evaluate_generator(self.data["test_generator"], steps=250, verbose=1)
            print("Loss / test: " + str(test_loss) + " and accuracy: " + str(acc_test))

        # Training is over
        self._training = False


if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
