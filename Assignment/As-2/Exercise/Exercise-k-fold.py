import numpy
import pandas as pd
from sklearn.model_selection import KFold

# Configurable constants
NUM_SPLITS = 100

file_path = '2.1-Exercise.csv'

data = pd.read_csv(file_path)
data = data.to_numpy()

# Perform a K-Fold split and print results
kfold = KFold(n_splits=NUM_SPLITS)
split_data = kfold.split(data)

print("""\
The K-Fold method works by splitting off 'folds' of test data until every point has been used for testing.

The following output shows the result of splitting some sample data.
A bar displaying the current train-test split as well as the actual data points are displayed for each split.
In the bar, "-" is a training point and "T" is a test point.
""")

print("Data:\n{}\n".format(data))
print('K-Fold split (with n_splits = {}):\n'.format(NUM_SPLITS))

for train, test in split_data:
    output_train = ''
    output_test = ''

    bar = ["-"] * (len(train) + len(test))

    # Build our output for display from the resulting split
    for i in train:
        output_train = "{}({}: {}) ".format(output_train, i, data[i])

    for i in test:
        bar[i] = "T"
        output_test = "{}({}: {}) ".format(output_test, i, data[i])

    print("[ {} ]".format(" ".join(bar)))
    print("Train: {}".format(output_train))
    print("Test:  {}\n".format(output_test))
