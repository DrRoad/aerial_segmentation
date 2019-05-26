"""
This file contains the main method of this program.
"""
try:
    from sys import version_info
    if version_info < (3, 6):
        exit("ERROR: you must use at least python 3.6")

    import sys
    from ui import MainWindow
    # Qt 5 imports
    from PyQt5.QtWidgets import QApplication
except ImportError as err:
    exit("{}: {}".format(__file__, err))


def main():
    """
    Main method.
    """
    # Create a QApplication instance
    app = QApplication(sys.argv)
    # Set its name
    app.setApplicationDisplayName("Segment images")
    # Create the main window
    my_window = MainWindow(950, 750)
    # Put this window in the application
    app.setActiveWindow(my_window)
    # Start the application
    exit(app.exec_())


if __name__ == "__main__":
    main()
