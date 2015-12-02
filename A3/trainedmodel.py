import scipy.io as spio
import numpy as np
from skimage import io
import time
from sklearn import cross_validation, datasets
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


from pylab import *

def main():
    labeled_images_data = spio.loadmat("labeled_images.mat")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))

    train_data, valid_data, train_targets, valid_targets, train_ident, valid_ident = splitSet(faces, labels, identities, 0.30)
    #best one is 65 with 0.61 rate
    for i in range(200):
        if i != 0:
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(train_data, train_targets)
            predictions = model.predict(valid_data)
            print(i)
            print(accuracy_score(predictions, valid_targets))



    return

def splitSet(data, targets, identities, validRatio):
    "Takes a set of data and returns a validation set and training set"
    number = np.around(len(data)*validRatio)
    borderIdent1 = identities[number]
    borderIdent2 = identities[number+1]
    if borderIdent1!=borderIdent2:
        print("Good split")
    if borderIdent1==-1 or borderIdent2==-1:
        print("Good split because one or both do not have an identity")
    else:
        print("Bad split")
    train_data = data[number:]
    valid_data = data[:number]
    train_ident = identities[number:]
    valid_ident =identities[:number]
    train_targ =  targets[number:].squeeze()
    valid_targ = targets[:number].squeeze()

    #cross_validation.train_test_split(data, targets, test_size=0.4,random_state=0)
    return train_data, valid_data, train_targ, valid_targ, train_ident, valid_ident


if __name__ == "__main__":
    main()