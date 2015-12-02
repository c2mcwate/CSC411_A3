import scipy.io as spio
import numpy as np
from skimage import io
import time
from sklearn import cross_validation, datasets, neighbors


def main():
    labeled_images_data = spio.loadmat("labeled_images.mat")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)

    train_data, valid_data, train_targets, valid_targets, train_ident, valid_ident = splitSet(faces, labels, identities, 0.30)

    return

def splitSet(data, targets, identities, validRatio):
    "Takes a set of data and returns a validation set and training set"
    number = np.around(len(data)*(1-validRatio))
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
    train_targ = targets[number:]
    valid_targ = targets[:number]
    #cross_validation.train_test_split(data, targets, test_size=0.4,random_state=0)
    return train_data, valid_data, train_targ, valid_targ, train_ident, valid_ident


if __name__ == "__main__":
    main()