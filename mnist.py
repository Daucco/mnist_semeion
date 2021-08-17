import gzip as gz
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Models
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

TEST_FILE = "test.gz"
TEST_LABELS = "test_labels.gz"
TRAIN_FILE = "train.gz"
TRAIN_LABELS = "train_labels.gz"

# Simple dataset loader. Can be use with any set
def mnist_loader(im, lb, flat=False):
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
    
    return (images, labels)

"""
parser = argparse.ArgumentParser(description="MNIST ds parser")
parser.add_argument("-d", "--data", type=str, help="Data folder with datasets. Expects train and test with labels.", required=True)
args = parser.parse_args()

datapath = args.data
"""

datapath = "dataset"

# Resolves paths
trainpath = str(pl.Path(datapath).joinpath("train.gz"))
trainpath_labels = str(pl.Path(datapath).joinpath("train_labels.gz"))
testpath = str(pl.Path(datapath).joinpath("test.gz"))
testpath_labels = str(pl.Path(datapath).joinpath("test_labels.gz"))

# Parses sets
train_imgs, train_labels = mnist_loader(trainpath, trainpath_labels, flat=True)
test_imgs, test_labels = mnist_loader(testpath, testpath_labels, True)

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
"""

# Slices train set (dummy only; reduces set size)
MAX_IMGS = 100
train_imgs = train_imgs[:MAX_IMGS]
train_labels = train_labels[:MAX_IMGS]

print("Training...")

##################################################################
################################# Trains svm classifier

# NOTE: (Opt.) Use gird search to automatically train multiple params and take the best result
grid_params = {
    "C": [.01, .1, 1, 10],
    "gamma": [.001, .01, .1],
    "kernel": ["linear", "rbf"]
}
grid_svc = GridSearchCV(svm.SVC(), grid_params)
grid_svc = grid_svc.fit(train_imgs, train_labels)
print("Best SVC params: %s" % grid_svc.best_params_)

model_svc = svm.SVC(gamma=grid_svc.best_params_["gamma"], kernel=grid_svc.best_params_["kernel"], C=grid_svc.best_params_["C"])
model_svc.fit(train_imgs, train_labels)

# Tests model
predicted = model_svc.predict(test_imgs)
print("Report of %s:\n%s" % (model_svc, metrics.classification_report(test_labels, predicted)))

##################################################################
################################# Trains random forest classifier
"""
model_rf = RandomForestClassifier(max_depth=2, random_state=0)
model_rf.fit(train_imgs, train_labels)

# Tests model
predicted = model_svc.predict(test_imgs)
print("Report of %s:\n%s" % (model_rf, metrics.classification_report(test_labels, predicted)))
"""

# TODO: Load semeion dataset and use the trained model with its test set

print("Done! :D")