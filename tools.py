import numpy as np
import pandas as pd


# load the dataset
def get_dataset(name):
    """from name return (X_train, y_train, X_test, y_test) if exist otherwise None for missing
    """
    # check the dataset name: mnist, mnist-fashion, macosko
    import sys

    if (name == "fashion-mnist"):
        sys.path.insert(0, "./datasets/fashion-mnist/")
        from utils import mnist_reader
        X_train, y_train = mnist_reader.load_mnist('./datasets/fashion-mnist/data/fashion', kind='train')
        X_test, y_test = mnist_reader.load_mnist('./datasets/fashion-mnist/data/fashion', kind='t10k')

    elif (name == "mnist-70k"):
        mnist = pd.read_csv('./datasets/mnist_784.csv', dtype=np.uint8)

        X_train = np.ascontiguousarray(mnist.iloc[:, :-1].values)
        y_train = np.ascontiguousarray(mnist.iloc[:, -1].values)
        X_test, y_test =  None, None

    elif (name == "mnist-2k"):
        from sklearn import datasets
        digits = datasets.load_digits()
        X_train = digits['data']
        y_train = digits['target']
        X_test, y_test =  None, None

    elif (name == "macosko"):
        # TBD: share the file that was made with opentsne notebook
        pass

    else:
        print('unknown dataset - exiting')
        sys.exit(0)

    return (X_train, y_train, X_test, y_test)
