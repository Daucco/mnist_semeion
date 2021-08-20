import numpy as np
import argparse
import pathlib as pl
import matplotlib.pyplot as plt
import sys

# Local
from ms_common import normalizeImgSet, cropImgToContent, fitImgToShape
from ms_common import dumpPickle, loadPickle
from ms_common import FIT_IMG_MODE_PERFECT as FIT_PERFECT
from ms_common import FIT_IMG_MODE_EXPAND as FIT_EXPAND

# Configs
SEMEION_IMG_SHAPE = (16, 16)
TARGET_IMG_SHAPE = (16, 16)
FITMODE = FIT_EXPAND

# Aux method: Transform a tensor of one-hot entries into a tensor of ints
# ex: [[0, 0, 1, 0], [0, 1, 0, 0]] -> [2, 1]
# PARAMS:
#   oh_list: Nested list of one-hot encoded numbers (0-9)
def onehotToInt(oh_list):
    oh_tensor = np.array(oh_list)
    int_tensor = np.argmax(oh_tensor, axis=1).astype(dtype=np.uint8)
    
    return int_tensor

# Simple semeion loader. Expects a single data file path (d)
def semeion_loader(d):
    images = []
    labels = []

    with open(d, "r") as data_fd:
        for line in data_fd.readlines():
            # Each line encodes an image
            line_data = line.split(" ")

            # Label is encoded in one-hot format within the last 10 digits of the line
            # Strips ending \n
            images.append(line_data[:-11])
            labels.append(line_data[-11:-1])

    # Once every item in the data file has been iterated, transforms the data into numpy objects
    images = np.array(images, dtype=np.float32).astype(dtype=np.uint8)

    # Reshapes images tensor. Each entry represents a 2D-image
    images = np.reshape(images, (-1, *SEMEION_IMG_SHAPE))

    labels = onehotToInt(labels)

    return (images, labels)

# Semeion images preprocessing. Adapts dataset content and shape
# Normalizes, crops and reshapes each image as required
def semeion_proc(images, norm=True, crop=True, fit=FITMODE, shape=TARGET_IMG_SHAPE, flat=False, verbose=True):
    if norm:
        images = normalizeImgSet(images, max_reference=1)

    proc_images = []
    nimg = len(images)
    for i, im in enumerate(images):
        if verbose:
            print("Processing image %d out of %d" % (i+1, nimg), end="\r", flush=True)

        if crop:
            im = cropImgToContent(im)

        im_fitReshape = fitImgToShape(im, shape, fit)
        proc_images.append(im_fitReshape)

    if verbose:
        sys.stdout.write("\033[K")  # Completely clears previous line at output
        print("Processed %d images!" % nimg)

    # Converts to numpy array
    images = np.array(proc_images)

    if flat:
        images = np.reshape(images, (nimg, -1))

    # Shorthand:
    #images = np.reshape(np.array([fitImgToShape(cropImgToContent(i), TARGET_IMG_SHAPE, FIT_PERFECT) for i in images]), (len(images), -1))

    return images

# This chunk only runs if this is the script being executed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semeion ds parser")
    parser.add_argument("-d", "--data", type=str, help="Data folder with datasets. A single .data file.", required=True)
    parser.add_argument("-o", "--out", type=str, help="Output folder to save the processed dataset.", required=True)
    args = parser.parse_args()

    datapath = args.data
    opath = args.out

    #datapath = "dataset_semeion"
    #opath = "dataset_semeion/proc"

    # Dummy; runs onehotToInt method
    """
    dummy_oh = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ]

    dummy_ohInt = onehotToInt(dummy_oh)
    print("Dummy; trying onehotToInt:\n%s" % str(dummy_ohInt))
    exit()
    """

    # Resolves paths
    datapath = str(pl.Path(datapath).joinpath("semeion.data"))

    # Parses sets
    print("Parsing SEMEION set...")
    images, labels = semeion_loader(datapath)

    # Image preprocessing: Normalizes image set, crops content, reshapes and flattens
    print("Preprocessing images...")
    images = semeion_proc(images, norm=True, crop=True, flat=True)

    # Saves resulting images and labels objects (pickle)
    print("Saving preprocessed data...")
    dumpPickle(images, "images", opath)
    dumpPickle(labels, "labels", opath)

    # Dummy; reloads pickles to check everything is golden
    images = loadPickle(str(pl.Path(opath).joinpath("images.pkl")))
    labels = loadPickle(str(pl.Path(opath).joinpath("labels.pkl")))

    # Dummy; shows image. RUN THISN ONLY IF THE IMAGES WEREN'T FLATTENED!
    """
    dummy_index = 77
    dummy_img = images[dummy_index]
    dummy_label = labels[dummy_index]
    print("min max: (%0.2f, %0.2f)" % (np.min(dummy_img), np.max(dummy_img)))
    print("Label is %d" % dummy_label)
    plt.gray()
    plt.imshow(np.asarray(dummy_img))
    plt.show()

    # TODO: Load classification model and try it with the set
    #   Might need to:
    #       * Create Train/test splits
    #       * Preprocess both, the mnist and semeion sets to enhance model compatibility (play with cropImgToContent and fitImgToShape)
    """

    print("All done! :D")