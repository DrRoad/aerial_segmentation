"""
Default neural network model. It helps to create a model using the Keras API.
"""
try:
    from os import makedirs
    from os.path import exists, join

    # Importing required Keras modules containing model and layers
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Concatenate, Activation, concatenate
    from keras.layers.normalization import BatchNormalization
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.optimizers import Adam, RMSprop, SGD

    import numpy as np

    from datetime import datetime

    from keras.models import model_from_json

    from .layers.pooling import MaxUnpooling2D, MaxPoolingWithArgmax2D
except ImportError as err:
    exit("{}: {}".format(__file__, err))


class NNModel(object):
    """
    Neural network model.
    """

    # Different model types
    _MODEL_TYPES = {
        'sequential': Sequential(),
        'model'     : Model()
    }

    # Constructor
    def __init__(self, model_type="model", data_to_process="buildings", model_name="model1"):
        """
        Initialization of the model.
        """
        self._model_type      = model_type
        self._data_to_process = data_to_process
        # Model training history
        self._history         = None
        # Training state is set to False
        self._training        = False
        # Model name
        self.model_name       = model_name
        # File extensions for data to predict
        self.FILE_EXTENSIONS  = list()
        # Log file
        self.logfile          = None
        # Initialize the main model
        self.__init_model()

    def __del__(self):
        """
        Destructor of the NNModel class.
        """
        if self.logfile is not None:
            self.logfile.close()

    def __init_model(self):
        """
        Initializes the main model.
        """
        if self._model_type not in self._MODEL_TYPES:
            raise NotImplementedError('Unknown model type: {}'.format(self._model_type))

        self._model = self._MODEL_TYPES[self._model_type]

    def add_layer(self, layer):
        """
        Adds a new layer to a sequential model.
        """
        self._model.add(layer)

    def rollback(self):
        """
        Re-initializes the model (~ rollback).
        """
        self.__init_model()

    # Getters
    def get_model_type(self):
        """
        Returns model name.
        """
        return self._model_type

    def get_model(self):
        """
        Returns an instance of the model.
        """
        return self._model

    def set_model(self, model):
        """
        Sets a new value to the current model.
        """
        self._model = model

    def get_history(self):
        """
        Returns the history of model training.
        """
        return self._history

    def is_training(self):
        """
        Returns the training state of the model. It returns True if the model
        is currently training, otherwise False.
        """
        return self._training

    # Setters
    def set_data_to_process(self, data_to_process):
        """
        Sets a new value to the type of data to process.
        """
        self._data_to_process = data_to_process

    def concatenate_extensions(self):
        """
        Concatenates all extensions in one sentence.
        """
        exts = ""
        for ext in self.FILE_EXTENSIONS:
            exts += "{} Files (*.{});;".format(ext.upper(), ext)
        exts += "All Files (*)"
        return exts

    def create_logfile(self):
        """
        Creates a log file for this model.
        """
        self.logfile = open("{}_{}".format(self._data_to_process, self.model_name) + datetime.now().strftime(
                       "_%d-%m-%y_%H-%M-%S") + ".txt", "w+")

    def save_model(self, basename="basename", folder="models"):
        """
        Saves a model.
        """
        if not exists(folder):
            makedirs(folder)  # Create a new directory if it doesn't exist

        architecture_file_path = basename + '.json'
        print('\t - Architecture of the neural network: ' + architecture_file_path)

        with open(join(folder, architecture_file_path), 'wt') as json_file:
            architecture = self._model.to_json()
            json_file.write(architecture)

        weights_file_path = join(folder, basename + '.hdf5')
        print('\t - Weights of synaptic connections: ' + weights_file_path)
        self._model.save(weights_file_path)

    def open_model(self, architecture_file_name, weights_file_name):
        """
        Opens an existing model.
        """
        if not exists(architecture_file_name):
            print("ERROR: " + architecture_file_name + " doesn't exist.")
            return
        elif architecture_file_name[-4:] != "json":
            print("ERROR: architecture file extension MUST BE json.")
            return

        if not exists(weights_file_name):
            print("ERROR: " + weights_file_name + " doesn't exist.")
            return
        elif weights_file_name[-4:] != "hdf5":
            print("ERROR: weights file extension MUST BE hdf5.")
            return

        json_file = open(architecture_file_name)
        architecture = json_file.read()
        json_file.close()
        # Create a model from a json file
        self._model = model_from_json(architecture,
                                      custom_objects={
                                          'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                                          'MaxUnpooling2D': MaxUnpooling2D
                                      })

        print(self._model)
        # Load weights into the model
        self._model.load_weights(weights_file_name)

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

    def load_files_to_predict(self, files):
        """
        Loads files to predict.
        """
        raise NotImplementedError("Please implement this method.")

    def predict_output(self):
        """
        Predicts an output for a given list of files/datas.
        """
        raise NotImplementedError("Please implement this method.")


if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
