# Simple testing script

import argparse
from sklearn import metrics

# Local
from ms_common import loadPickle

# This chunk only runs if this is the script being executed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("-dX", "--dataImages", type=str, help="Test dataset (pickle). Expects a tensor of flattened images", required=True)
    parser.add_argument("-dy", "--dataLabels", type=str, help="Test labels (pickle)", required=True)
    parser.add_argument("-m", "--model", type=str, help="Testing model (pickle)", required=True)
    args = parser.parse_args()

    datapath = args.dataImages
    labelspath = args.dataLabels
    modelpath = args.model

    #datapath = "dataset_semeion/proc/images.pkl"
    #labelspath = "dataset_semeion/proc/labels.pkl"
    #modelpath = "output_models/svc.pkl"

    # Loads test data (X and y)
    test_X = loadPickle(datapath)
    test_y = loadPickle(labelspath)

    # Loads model
    estimator = loadPickle(modelpath)

    # Runs test and gets results report
    predicted = estimator.predict(test_X)
    report = metrics.classification_report(test_y, predicted)
    print("TEST Report of %s:\n%s" % (estimator, report))