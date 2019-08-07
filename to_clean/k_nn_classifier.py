import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def K_NN_classifier(data, labels, K=1):

    split = sklearn.model_selection.ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
    id_train, id_test = next(split.split(data))

    nnc_train = data[id_train]
    nnc_test  = data[id_test]

    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(nnc_train, labels[id_train]) 

    y_pred = neigh.predict(nnc_test)
    y_true = labels[id_test]

    return accuracy_score(y_true, y_pred)
