import sys
import numpy as np
from matplotlib import pyplot
from Beams import*
from Parser import*


if __name__ == "__main__":

    parser = Parser.Parser(sys.argv[1])
        
    if parser['Beams']['do_beams']:
        main(parser)

