import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# set directory paths
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'data')

# read files
train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

# create train and test set
### read images and store as numpy arrays
temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
train_x = np.stack(temp)
train_x = train_x.reshape(-1, 784)

temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
test_x = np.stack(temp)
test_x = test_x.reshape(-1, 784)

train_y = train.label.values

# define model
clf = RandomForestClassifier(n_estimators = 50, n_jobs=3)

#print "5-fold accuracies: ", cross_val_score(clf, train_x, train_y, cv=5)

# train model
clf.fit(train_x, train_y);

# predict model
pred = clf.predict(test_x);

# save submission
sample_submission['filename'] = test.filename
sample_submission['label'] = pred
sample_submission.to_csv('sub01.csv', index=False)
