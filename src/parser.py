try:
    from os.path import basename
except ImportError as err:
    exit("{}: {}".format(__file__, err))


# This is a generic class. It could be interesting to add this one in 
# a global module with other generic functions/class.
class Parser(object):
    """
    An argument parser.
    """

    def __init__(self, argv, possible_args=["--help"], args=dict()):
        self.__argv = argv
        self.__possible_args = possible_args
        self.__args = args

    def get_argv(self):
        return self.__argv

    def get_possible_args(self):
        return self.__possible_args

    def get_args(self):
        """
        Returns a dictionary:
            - key  : arguments,
            - value: actions.
        """
        return self.__args

    def my_args(self):
        """
        Gets all of the arguments and values from the sys.argv list. Each argument/value couple must be write this way:
            [argument]=[value]
        """
        start = 0
        if self.__argv[0] == basename(__file__):
            start = 1

        for i in range(start, len(selg.__argv), 1):
            # Split the current argv to retrieve the argument and its value
            a = self.__argv[i].split("=")
            # If the length is not equal to 2 we must return an error
            if len(a) != 2:
                self.__raise_error("ERROR: you have to use args this way -> [argument]=[value]")
            # Get the argument and its value
            arg, val = a[0], a[1]
            # Is it a known argument ?
            if arg not in self.__possible_args:
                self.__raise_error("ERROR: unknown argument -> " + arg + "\nSee --help=args for all arguments.")
            # Store the key (argument) and its value
            self.__args[arg] = val

    @staticmethod
    def __raise_error(txt):
        # TODO : we MUST find a better way to handle errors (exit(txt) is not efficient)
        exit(txt)


if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
