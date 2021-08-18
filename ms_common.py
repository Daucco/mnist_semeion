# Common tools
import numpy as np
import cv2
import pickle as pk
import pathlib as pl
import matplotlib.pyplot as plt

# Globals
FIT_IMG_MODE_PERFECT = 0
FIT_IMG_MODE_EXPAND = 1

# Stores an object as a byte stream (pickle)
def dumpPickle(object, name, path, verbose=True):
    fullPath = pl.Path(path).joinpath("%s.pkl" % name)
    pk.dump(object, open(fullPath, "wb"))

    if verbose:
        print("<<< Object \"%s\" saved at \"%s\"" % ("%s.pkl" % name, path))

    # Returns full path of the saved object
    return fullPath

# Loads an object from a byte stream
def loadPickle(path, verbose=True):
    if verbose:
        print(">>> Loading object from \"%s\"" % path)

    with open(path, "rb") as object_dataStream:
        return pk.load(object_dataStream)

# Divides two elements rounding up
def divRoundUp(dividend, divisor):
    return int(dividend // divisor + (dividend % divisor > 0))

# Crops down an image given as a numpy tensor
# The result is the smallest subtensor that perfectly fits the image content
def cropImgToContent(image, verbose=False):

    # Resolves boundaries of content

    # Sums row-wise (collapses rows). Obtains a vector of shape (1, ncolums)
    image_rowSum = np.sum(image, axis=0)

    # NOTE: Need to explicitly take the first element [0] because by default np.where returns a 2D object,
    #   no matter the shape of the input tensor
    image_rowSum_nonZeroPos = np.where(image_rowSum > 0)[0]
    image_xmin = image_rowSum_nonZeroPos[0]
    image_xmax = image_rowSum_nonZeroPos[-1]

    # Sums col-wise (collapses columns). Obtains a vector of shape (nrows, 1)
    image_colSum = np.sum(image, axis=1)
    image_colSum_nonZeroPos = np.where(image_colSum > 0)[0]
    image_ymin = image_colSum_nonZeroPos[0]
    image_ymax = image_colSum_nonZeroPos[-1]

    if verbose:
        print("Image boundaries: (%s, %s)" % (str((image_ymin, image_ymax)), str((image_xmin, image_xmax))))

    # Returns cropped image as a subset of the original tensor (uses slicing)
    return image[image_ymin:(image_ymax + 1), image_xmin:(image_xmax + 1)]

# Reshapes a given image to fit a target shape
# PARAMS:
#   image: input image as a 2D tensor; shape: target shape as a 2-tuple
#   fitmode: controls how the image is reshaped
#       PERFECT: Keeps original aspect. Scales original image and fills remaining pixels with zeroes to fit the target shape
#       EXPAND: Disregards original aspect. Scales image to target shape
def fitImgToShape(image, shape, fitmode=FIT_IMG_MODE_PERFECT):
    # If perfect fit, adds empty pixels as needed before scaling
    if fitmode == FIT_IMG_MODE_PERFECT:
        image_shape = image.shape
        image_rows, image_columns = image_shape
        target_aspect = shape[0] / shape[1]
        image_aspect = image_rows / image_columns

        if image_aspect < target_aspect:
            # Input immage is proportionally shorter than the target shape
            #   Adds empty rows to make the image taller
            image_extendedShape_rows = divRoundUp(target_aspect, image_aspect) * image_rows
            image_extendedShape = (image_extendedShape_rows, image_columns)

            # Creates new tensor of the required shape and centers content into it (vertically)
            rowOffset = (image_extendedShape_rows - image_rows) // 2
            image_extended = np.zeros(image_extendedShape, dtype=image.dtype)
            image_extended[rowOffset:(image_rows + rowOffset)] = image
            image = image_extended


        elif image_aspect > target_aspect:
            # Input image is proportionally taller than the target shape
            #   Adds empty columns to make the image wider
            image_extendedShape_columns = divRoundUp(image_aspect, target_aspect) * image_columns
            image_extendedShape = (image_rows, image_extendedShape_columns)

            # Creates new tensor of the required shape and centers content into it (horizontally)
            colOffset = (image_extendedShape_columns - image_columns) // 2
            image_extended = np.zeros(image_extendedShape, dtype=image.dtype)
            image_extended[:, colOffset:(image_columns + colOffset)] = image
            image = image_extended

    # Scales image to target shape
    # NOTE: OpenCV expects the dimensions to be specified in (width, height) format (opposite to the dimensions of a numpy array!)
    # NOTE: Interpolation controls how the image is scaled (how new pixels are created or dropped). Use nearest if no new pixel value should appear in the scaled image
    #   IF USING ANY OTHER INTERPOLATION THAN NEAREST, ENSURE THE INPUT IMAGE IS OF FLOAT TYPE, OTHERWISE THE RESIZE METHOD DOES NOT WORK!
    #image = cv2.resize(image.astype(dtype=np.float32), shape[::-1], interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, shape[::-1], interpolation=cv2.INTER_NEAREST)
    
    return image

if __name__ == "__main__":
    
    # Dummy; tries cropImgToContent method
    dummy_img = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0]
        ]
    )
    dummy_img_crop = cropImgToContent(dummy_img, True)

    print("Dummy crop original (%s):\n%s\n---" % (str(dummy_img.shape), str(dummy_img)))
    print("Dummy crop cropped (%s):\n%s\n---" % (str(dummy_img_crop.shape), str(dummy_img_crop)))

    # Dummy; tries fitImgToShape method
    dummy_img_targetShape = (16, 16)
    dummy_img_fitPerfect = fitImgToShape(dummy_img_crop, dummy_img_targetShape)
    dummy_img_fitExpand = fitImgToShape(dummy_img_crop, dummy_img_targetShape, fitmode=FIT_IMG_MODE_EXPAND)

    print("Dummy crop fit perfect (%s):\n%s\n---" % (str(dummy_img_fitPerfect.shape), str(dummy_img_fitPerfect)))
    print("Dummy crop fit expand (%s):\n%s\n---" % (str(dummy_img_fitExpand.shape), str(dummy_img_fitExpand)))
    
    # Fancy visualization
    fig = plt.figure(figsize=(12, 3))
    plt.gray()
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.title.set_text("Dummy")
    plt.imshow(dummy_img)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.title.set_text("Crop")
    plt.imshow(dummy_img_crop)
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.title.set_text("Fit perfect")
    plt.imshow(dummy_img_fitPerfect)
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.title.set_text("Fit expand")
    plt.imshow(dummy_img_fitExpand)
    plt.show()