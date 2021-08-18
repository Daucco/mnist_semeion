import numpy as np
import argparse
import pathlib as pl
import matplotlib.pyplot as plt

SEMEION_IMG_SHAPE = (16, 16)

# Aux method: Transform a tensor of one-hot entries into a tensor of ints
# ex: [[0, 0, 1, 0], [0, 1, 0, 0]] -> [2, 1]
# PARAMS:
#   oh_list: Nested list of one-hot encoded numbers (0-9)
def onehotToInt(oh_list):
    oh_tensor = np.array(oh_list)
    int_tensor = np.argmax(oh_tensor, axis=1).astype(dtype=np.uint8)
    
    return int_tensor

# Simple semeion loader. Expects a single data file path (d)
def semeion_loader(d, flat=False):
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
    if not flat:
        images = np.reshape(images, (-1, *SEMEION_IMG_SHAPE))

    labels = onehotToInt(labels)

    return (images, labels)


# This chunk only runs if this is the script being executed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semeion ds parser")
    parser.add_argument("-d", "--data", type=str, help="Data folder with datasets. a single .data file.", required=True)
    args = parser.parse_args()

    datapath = args.data

    #datapath = "dataset_semeion"

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
    images, labels = semeion_loader(datapath)

    # Dummy; shows image
    dummy_index = 77
    dummy_img = images[dummy_index]
    dummy_label = labels[dummy_index]
    print("min max: (%d, %d)" % (int(np.min(dummy_img)), int(np.max(dummy_img))))
    print("Label is %d" % dummy_label)
    plt.gray()
    plt.imshow(np.asarray(dummy_img))
    plt.show()

    # TODO: Load classification model and try it with the set
    #   Might need to:
    #       * Create Train/test splits
    #       * Preprocess both, the mnist and semeion sets to enhance model compatibility (play with cropImgToContent and fitImgToShape)