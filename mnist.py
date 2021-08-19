import gzip as gz
import argparse
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score

# Local
from ms_common import normalizeImgSet, dumpPickle, loadPickle

# Models
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

TEST_FILE = "test.gz"
TEST_LABELS = "test_labels.gz"
TRAIN_FILE = "train.gz"
TRAIN_LABELS = "train_labels.gz"

# Applies intensity threshold to the image's pixels
def mnist_simpleThreshold(img_pixels, thresh):
    thresh_value = thresh * 254

    # Binary encoding: Only the pixels above threshold are non-zero
    images = np.where(img_pixels > thresh_value, 1, 0)

    return images

# Simple mnist loader. Can be use with any set
def mnist_loader(im, lb, flat=False, norm=False, thresh=-1):
    images = None
    labels = None

    # Loads data
    with gz.open(im, "r") as images_fd:
        # Retrieves dataset metainfo from header
        images_fd.read(4)
        nimgs = int.from_bytes(images_fd.read(4), "big")
        img_size = (int.from_bytes(images_fd.read(4), "big"), int.from_bytes(images_fd.read(4), "big"))

        # Reads remaining bytes containing the dataset
        buf_img = images_fd.read()
        images = np.frombuffer(buf_img, dtype=np.uint8)

        # Flattens data if required
        if flat:
            images = images.reshape((nimgs, -1))
        else:
            images = images.reshape((nimgs, *img_size))
            print("IMAGESSS")
            print(images)

    # Loads labels
    with gz.open(lb, "r") as labels_fd:

        # Retrieves dataset metainfo from header
        labels_fd.read(4)
        nlabels = int.from_bytes(labels_fd.read(4), "big")

        # Reads remaining bytes containing the dataset
        buf_lab = labels_fd.read()
        labels = np.frombuffer(buf_lab, dtype=np.uint8)
        #labels = labels.reshape((nlabels))

    # Applies threshold to the images if required
    if 0.0 <= thresh <= 1.0:
        images = mnist_simpleThreshold(images, thresh)

    # Normalizes images if required
    if norm:
        #images = normalizeImgSet(images, max_reference=255) # See mnist_simpleThreshold output
        images = normalizeImgSet(images)
    
    return (images, labels)

# This chunk only runs if this is the script being executed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST ds parser")
    parser.add_argument("-d", "--data", type=str, help="Data folder with datasets. Expects train and test with labels.", required=True)
    parser.add_argument("-s", "--modelSave", type=str, help="Resulting models savepath folder.", required=True)
    args = parser.parse_args()

    datapath = args.data
    savepath = args.modelSave

    #datapath = "dataset_mnist"
    #savepath = "output_models"

    # Resolves paths
    trainpath = str(pl.Path(datapath).joinpath("train.gz"))
    trainpath_labels = str(pl.Path(datapath).joinpath("train_labels.gz"))
    testpath = str(pl.Path(datapath).joinpath("test.gz"))
    testpath_labels = str(pl.Path(datapath).joinpath("test_labels.gz"))

    # Parses sets
    # NOTE RAWR: Try and see what happens when toggling normalization (norm=True/False). Also fiddle a bit with threshold
    #train_imgs, train_labels = mnist_loader(trainpath, trainpath_labels, flat=False, thresh=0.5)
    train_imgs, train_labels = mnist_loader(trainpath, trainpath_labels, flat=True, norm=True)
    test_imgs, test_labels = mnist_loader(testpath, testpath_labels, True, norm=True)

    """
    # NOTE: Remove this
    plt.gray()
    plt.imshow(train_imgs[77])
    plt.show()
    exit()

    test_imgs_se = loadPickle("dataset_semeion/proc/images.pkl")
    test_labels_se = loadPickle("dataset_semeion/proc/labels.pkl")

    print("SE:\n%s" % str(test_imgs_se[77]))
    print("SE label:\n%s" % str(test_labels_se[77]))
    print("(%d, %d)" % (len(test_imgs_se), len(test_labels_se)))
    print("M:\n%s" % str(test_imgs[77]))
    print("M label:\n%s" % str(test_labels[77]))
    print("(%d, %d)" % (len(test_imgs), len(test_labels)))

    test_imgs = test_imgs[77:78]
    test_labels = test_labels[77:78]
    """

    # Dummy; shows image
    """
    dummy_index = 2222
    dummy_img = train_imgs[dummy_index]
    dummy_label = train_labels[dummy_index]
    print("min max: (%d, %d)" % (int(np.min(dummy_img)), int(np.max(dummy_img))))
    print("Label is %d" % dummy_label)
    plt.gray()
    plt.imshow(np.asarray(dummy_img))
    plt.show()
    exit()
    """

    # Slices train set (dummy only; reduces set size)
    MAX_IMGS = 100
    train_imgs = train_imgs[:MAX_IMGS]
    train_labels = train_labels[:MAX_IMGS]

    """
    print(np.unique(train_labels))
    print(np.unique(test_labels))
    """

    print("Training...")

    ##################################################################
    ################################# Trains svm classifier

    # NOTE: (Opt.) Use gird search to automatically train multiple params and take the best result
    # See magic: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    #   https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee
    grid_params = {
        "C": [.01, .1, 1, 10],
        "gamma": [.001, .01, .1],
        "kernel": ["linear", "rbf"]
    }
    grid_svc = GridSearchCV(estimator=svm.SVC(), param_grid=grid_params, scoring="accuracy", cv=5, refit=True)
    grid_svc = grid_svc.fit(train_imgs, train_labels)
    print("Best SVC params: %s" % grid_svc.best_params_)

    # NOTE: No need to train the estimator again if setting "refit" on grid search. Just pick it up from the grid search output
    #model_svc = svm.SVC(gamma=grid_svc.best_params_["gamma"], kernel=grid_svc.best_params_["kernel"], C=grid_svc.best_params_["C"])
    model_svc = grid_svc.best_estimator_
    #model_svc.fit(train_imgs, train_labels)

    """
    # Alternatively, estimator tuning through cv can be done manually
    model_svc = svm.SVC(C=.1, gamma=.01, kernel="linear")
    score = cross_val_score(estimator=model_svc, X=train_imgs, y=train_labels, scoring="accuracy", cv=5)
    
    # NOTE: global score is resolved from the score of each individual fold, by computing the mean and the std
    #   Namely, this is the AUC: https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it
    global_score = score.mean()
    global_score_deviation = score.std()
    print("TRAIN AUC: %0.2f (+- %0.2f)" % (global_score, global_score_deviation))

    # NOTE: Perform cv scoring for every configuration (and every estimator), then compare scores, pick the best params and fit the final model:
    model_svc = svm.SVC(C=.1, gamma=.01, kernel="linear")
    model_svc.fit(train_imgs, train_labels)
    """
    

    # Saves model to output folder
    dumpPath_svc = dumpPickle(model_svc, name="svc", path=savepath)

    # Dummy; loads the model just saved to use it in testing
    model_svc = loadPickle(dumpPath_svc)

    # Tests model
    predicted = model_svc.predict(test_imgs)
    print("TEST Report of %s:\n%s" % (model_svc, metrics.classification_report(test_labels, predicted)))

    ##################################################################
    ################################# Trains random forest classifier
    """
    model_rf = RandomForestClassifier(max_depth=2, random_state=0)
    model_rf.fit(train_imgs, train_labels)

    # Tests model
    predicted = model_svc.predict(test_imgs)
    print("Report of %s:\n%s" % (model_rf, metrics.classification_report(test_labels, predicted)))
    """
    print("Done! :D")