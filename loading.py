import sys
from data_manipulation import load

if __name__ == '__main__':
    # Load the model
    model = sys.argv[1]
    load(sys.argv[1])

