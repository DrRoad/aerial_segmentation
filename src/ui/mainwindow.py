"""
Main window.
"""
try:
    import sys
    from datetime import datetime

    from os.path import join, basename, splitext
    from os import makedirs, getpid
    import psutil

    # Design
    from .design import Ui_MainWindow

    from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
    from PyQt5.QtGui import QTextCursor
    from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QThread

    import tensorflow as tf
    import numpy as np

    import time

    # Plots
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

    from PIL import Image, ImageQt

    # Importing keras
    from keras import backend as K

    # In our case we're using the GPU which is faster than the CPU
    GPU = True

    if GPU:
        num_CPU = 1
        num_GPU = 1
    else:
        num_CPU = 1
        num_GPU = 0
    # Create a new Tensorflow configuration
    config = tf.ConfigProto(intra_op_parallelism_threads=0,
                            inter_op_parallelism_threads=0,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU},
                            gpu_options=tf.GPUOptions(allow_growth=True)
                            )
    # Create a new session with this configuration
    session = tf.Session(config=config)
    # Set the Keras session with the one from Tensorflow
    K.set_session(session)

    # Import CNN models
    from nnmodels import available_models
    from nnmodels import SegNet, VNet
except ImportError as err:
    exit("{}: {}".format(__file__, err))


tf_session = K.get_session()
tf_graph = tf.get_default_graph()


class ThreadCreateModel(QThread):
    """
    TODO: documentation.
    """
    threadDone = pyqtSignal(bool)

    def __init__(self, mw, action="create_model"):
        super().__init__()
        # Main window instance
        self.__mw = mw

        self.action = action

        self.threadDone.connect(self.__mw.done)

    def __del__(self):
        self.wait()

    def run(self):
        """
        Main function -> it runs the current thread until its work is done.
        """
        if self.action == "create_model":
            self.create_model()
        elif self.action == "create_models":
            self.create_models()
        else:
            raise ValueError("ERROR: {} is an unknown action.".format(self.action))

    def create_model(self):
        """
        Creates one model.
        """
        with tf_session.as_default():
            with tf_graph.as_default():
                self.__mw.get_model().learn()

                self.threadDone.emit(True)

    def create_models(self):
        """
        Creates at least two models.
        """
        return

    def stop(self):
        """
        Stop the current thread.
        """
        self.terminate()
        self.threadDone.emit(False)


class EmittingStream(QObject):
    """
    TODO: documentation.
    """
    textWritten = pyqtSignal(str)

    def write(self, text):
        """
        Emits a signal to write in a QTextEdit.

        :param text: the text to write
        """
        self.textWritten.emit(str(text))

    def flush(self):
        """
        Necessary to avoid this error -> AttributeError: 'EmittingStream' object has no attribute 'flush'
        """
        pass


class MainWindow(QMainWindow):
    """
    Main window using PyQt5 library.
    """

    def __init__(self, x, y, title="Main Window"):
        """
        Class constructor.
        """
        super().__init__()

        # Title
        self.setWindowTitle(title)

        # Window size
        self.__x, self.__y, = x, y
        # Default data to process
        self.__data_to_process = "roads"
        # Inform the program that multiple learning is used (and so threading is not enabled)
        self.multiple_learning = False
        # Inform the program that it is able to train a new neural network or not
        self.__is_training_enabled = True
        # Create a variable to prevent the use of an uninitialized model
        self.__is_model_loaded = False
        # Default model
        self.DefaultModel = VNet

        # Install the custom output stream
        sys.stdout = EmittingStream(textWritten=self.normal_output_written)

        # Initialize the model
        self.__init_model()
        # Initialize the main UI
        self.__init_ui()

    def __del__(self):
        """
        Class destructor.
        """
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def get_model(self):
        """
        Returns an instance of the current model.
        """
        return self.model

    def __init_model(self):
        with tf_session.as_default():
            with tf_graph.as_default():
                # Create a variable for neural network model
                self.model = self.DefaultModel()
                # The data to predict
                self.__data_to_predict = list()

    def __init_ui(self):
        """
        Initialization of the main window UI.
        """
        # Create the main UI
        self.__ui = Ui_MainWindow()
        self.__ui.setupUi(self)
        # Create plots
        self.__create_plots()
        # Create connections
        self.__create_connections()
        # Show main window
        self.show()

    def __create_plots(self):
        """
        Creates plots to show model learning performance.
        """
        # A figure instance to plot on
        self.__figure = plt.figure()

        # This is the Canvas Widget that displays the 'figure'
        # It takes the 'figure' instance as a parameter to __init__
        self.__canvas = FigureCanvas(self.__figure)

        # This is the Navigation Widget
        # It takes the Canvas widget and a parent as parameters to __init__
        self.__toolbar = NavigationToolbar(self.__canvas, self)

        self.__ui.plotLayout.addWidget(self.__toolbar)
        self.__ui.plotLayout.addWidget(self.__canvas)

    def __create_connections(self):
        """
        Creates connections between actions/buttons/etc and methods.
        """
        # Actions
        self.__ui.exitAct.triggered.connect(self.close)
        self.__ui.openAct.triggered.connect(self.btn_open_file)
        # Buttons
        self.__ui.createModelBtn.clicked.connect(self.btn_create_model)
        self.__ui.openModelBtn.clicked.connect(self.btn_open_model)
        self.__ui.predictValueBtn.clicked.connect(self.btn_predict_value)
        self.__ui.plotBtn.clicked.connect(self.btn_plot)
        self.__ui.multipleLearningBtn.clicked.connect(self.btn_several_learnings)
        # Radio buttons
        self.__ui.roadsRadioBtn.clicked.connect(lambda: self.set_model("roads"))
        self.__ui.digitsRadioBtn.clicked.connect(lambda: self.set_model("digits"))
        self.__ui.animalsRadioBtn.clicked.connect(lambda: self.set_model("animals"))
        self.__ui.aerialBuildingsRadioBtn.clicked.connect(lambda: self.set_model("aerial_buildings"))
        self.__ui.aerialRoadsRadioBtn.clicked.connect(lambda: self.set_model("aerial_roads"))
        self.__ui.kidneysRadioBtn.clicked.connect(lambda: self.set_model("kidneys"))

    def closeEvent(self, event):
        """
        Before closing the main window we ask user agreement.
        """
        reply = QMessageBox.question(self,
                                     "Message",
                                     "Are you sure you want to quit ?",
                                     QMessageBox.Yes,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    @pyqtSlot(str)
    def normal_output_written(self, text):
        """
        Appends text to the QTextEdit.
        """
        # If we're currently generating a new neural network model
        if self.model.is_training():
            txt_split = text.split(" ")
            # Get the epoch and show it in the progress bar
            if txt_split[0] == "Epoch":
                step = txt_split[1].split("/")
                actual_step = step[0]
                max_step = step[1]
                self.__ui.progressBar.setValue(
                    ((int(actual_step) / int(max_step)) * 100) % 101)  # Modulo 101 -> [0;100] and not [0;100[

        if self.__ui.loggerCheckBox.isChecked() and any(t in text for t in ("val_loss", "Epoch")):
            self.model.logfile.write(text)
            self.model.logfile.flush()

        if self.__ui.useConsoleCheckBox.isChecked():
            cursor = self.__ui.textEdit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)
            self.__ui.textEdit.setTextCursor(cursor)
            self.__ui.textEdit.ensureCursorVisible()
        else:
            print(text, file=sys.__stdout__)

    @pyqtSlot()
    def btn_open_file(self):
        """
        Opens a list of files.
        """
        # Set options
        options = QFileDialog.Options()  # No option
        # Get file path
        filenames, _ = QFileDialog.getOpenFileNames(self, "Open File", "", self.model.concatenate_extensions(),
                                                    options=options)
        # Create an empty list
        files = list()
        for filename in filenames:
            if filename.endswith(tuple(self.model.FILE_EXTENSIONS)):
                # Add this file in the files to process
                files.append(filename)
            else:
                print("ERROR: wrong file extension.")

        # Check the length of this list to not write a void list in
        # the MainWindow.__data_to_predict list
        if len(files) != 0:
            # Copy these values in the MainWindow.__data_to_predict list
            self.__data_to_predict = files

    @pyqtSlot()
    def btn_open_model(self):
        """
        Opens an existing model.
        """
        # Set options
        options = QFileDialog.Options()
        # Get file path
        model_filename, _ = QFileDialog.getOpenFileName(self,
                                                        "Open Architecture File",
                                                        "",
                                                        "JSON Files (*.json);;HDF5 Files (*.hdf5);;All Files (*)",
                                                        options=options)
        # Avoid empty file name
        if len(model_filename) != 0:
            if model_filename.endswith(tuple(["json", "hdf5"])):
                try:
                    # Get the type of the model from its filename
                    self.model = available_models[splitext(basename(model_filename))[0].split("_")[1]]

                    # Extract the name
                    model_filename = model_filename.split(".")[0]
                    # Open the model
                    self.model.open_model(model_filename + ".json", model_filename + ".hdf5")
                    # Allow the use of the model
                    self.__is_model_loaded = True

                    try:
                        filename = basename(model_filename)
                        self.__ui.currentModel.setText(filename)
                        self.__data_to_process = filename.split("_")[0]
                        self.model.set_data_to_process(self.__data_to_process)
                    except IndexError:
                        self.__ui.currentModel.setText(model_filename)
                except KeyError:
                    print("ERROR: unknown model -> {}".format(splitext(basename(model_filename))[0].split("_")[1]))
            else:
                print("ERROR: wrong file extension.")

    @pyqtSlot()
    def btn_create_model(self):
        """
        Creates a NN model.
        """
        if not self.__is_training_enabled:
            return

        if len([p.info for p in psutil.process_iter(attrs=['pid', 'name']) if 'python' in p.info['name']]) != 1:
            print("Hold on, wait a little buddy...", end=' ', flush=True)
            # Wait during 4 hours
            time.sleep(60 * 60 * 4)
            print("Let's go!")
            print("First of all, kill the previous python processes...", end=' ', flush=True)
            # Get my own pid
            my_pid = getpid()
            # Get all python processes
            py_processes = [p.info for p in psutil.process_iter(attrs=['pid', 'name']) if 'python' in p.info['name']]
            for process in py_processes:
                if process['pid'] != my_pid:
                    p = psutil.Process(process['pid'])
                    p.kill()
            time.sleep(60)
            print("OK, let's train!")

        # To avoid ValueError error we have to do a rollback of the model
        self.model.rollback()
        # Create all layers
        self.model.create_layers()

        if self.multiple_learning:
            self.model.learn()
        else:
            # Compile, fit and evaluate the model in a new thread
            self.thread = ThreadCreateModel(self)
            self.thread.setTerminationEnabled()
            # Start the thread
            self.thread.start()
        # self.__ui.stopModelCreationBtn.setEnabled(True)
        # if not self.__isStopModelCreationBtnConnected:
        #    self.__ui.stopModelCreationBtn.clicked.connect(self.__thread.stop)
        #    self.__isStopModelCreationBtnConnected = True

    @pyqtSlot()
    def btn_predict_value(self):
        """
        Predicts the segmentation of a list of images. Before
        calling this method you have to load at least one image
        with the method MainWindow.openFile().
        """
        if self.__is_model_loaded:
            if len(self.__data_to_predict) != 0:
                # Load these files in the model
                self.model.load_files_to_predict(self.__data_to_predict)
                # Predict the output
                self.model.predict_output()
            else:
                print("ERROR: please, open an image to predict.")
        else:
            print("ERROR: please, create or open a model.")

    @pyqtSlot()
    def btn_several_learnings(self):
        """
        Train multiple neural networks, it takes a little break
        between each one.
        """
        if not self.__is_training_enabled:
            return

        reply = QMessageBox.question(self,
                                     "Multiple learning",
                                     "Multiple learning ?",
                                     QMessageBox.Yes,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.multiple_learning = True

            # Compile, fit and evaluate the model in a new thread
            self.thread = ThreadCreateModel(self, "create_models")
            self.thread.setTerminationEnabled()
            # Start the thread
            self.thread.start()

    @pyqtSlot()
    def btn_plot(self):
        """
        Plots model accuracy and loss.
        """
        # Random data
        if self.model is not None and self.model.get_history() is not None:
            history = self.model.get_history()
            print(history.history.keys())
            keys = list(history.history.keys())
            # Training accuracy and loss
            x_acc = history.history[keys[3]]
            x_loss = history.history[keys[2]]
            # Validation accuracy and loss
            x_val_acc = history.history[keys[1]]
            x_val_loss = history.history[keys[0]]

            # Instead of ax.hold(False)
            self.__figure.clear()

            # Create 2 axes
            ax1, ax2 = self.__figure.subplots(1, 2)

            # Plot datas
            # Accuracy
            train_legend, = ax1.plot(x_acc, label="training accuracy")
            val_legend, = ax1.plot(x_val_acc, label="validation accuracy")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("accuracy")
            ax1.set_title("Training and validation accuracy")
            # Loss
            ax2.plot(x_loss, label="training loss")
            ax2.plot(x_val_loss, label="validation loss")
            ax2.set_xlabel("epoch")
            ax2.set_ylabel("loss")
            ax2.set_title("Training and validation loss")

            # Legend
            self.__figure.legend(handles=(train_legend, val_legend), labels=('training', 'validation'), loc='best')

            # Create a new directory if it doesn't exist
            folder = "models/figures"
            makedirs(folder, exist_ok=True)
            # Save the figure
            filename = self.saved_model_filename + ".png"
            self.__figure.savefig(join(folder, filename))

            # Refresh canvas
            self.__canvas.draw()
        else:
            print("WARNING: nothing to plot (model and/or model.history is/are None).")

    @pyqtSlot(bool)
    def done(self, ret):
        """
        Informs the user of the end of the model training.
        """
        self.__ui.progressBar.setValue(0)

        if ret:
            self.saved_model_filename = "{}_{}".format(self.__data_to_process, self.get_model().model_name) + datetime.now().strftime("_%d-%m-%y_%H-%M-%S")
            # Save the model with the given name
            self.model.save_model(self.saved_model_filename)
            print("Model saved.")

            # Allow the use of the model
            self.__is_model_loaded = True

            self.__ui.currentModel.setText(self.saved_model_filename)

        print("DONE")

    @pyqtSlot(str)
    def set_model(self, radio_btn):
        """
        Thanks to the selected radio button, instantiate a new neural network model.
        """
        self.__data_to_process = radio_btn
        if radio_btn == "aerial_buildings":
            self.model = self.DefaultModel()


if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
