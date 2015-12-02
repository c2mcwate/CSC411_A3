import scipy.io as spio
import numpy as np
from skimage import io
import time
from sklearn import cross_validation, datasets
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier
from pylab import *

def knn():
    labeled_images_data = spio.loadmat("labeled_images.mat")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))
    train_data, test_data, train_targets, test_targets, train_ident, target_ident = splitSet(faces, labels, identities, 0.3)
    #best one is 65 with 0.61 rate
    kf = KFold(len(train_data), n_folds=4)

    k_values = [7,22,37,50,64]
    k_values_class = np.zeros(5)
    for i in range(len(k_values)):
        k=0
        cross_val_accuracy = np.zeros(4)
        for train_index, test_index in kf:
              #print("TRAIN:", train_index, "TEST:", test_index)
              train_data_fold, valid_data_fold = train_data[train_index], train_data[test_index]
              train_targets_fold, valid_targets_fold = train_targets[train_index], train_targets[test_index]
              model = KNeighborsClassifier(n_neighbors=k_values[i])
              model.fit(train_data_fold, train_targets_fold)

              predictions = model.predict(valid_data_fold)
              if k<=3:
                cross_val_accuracy[k]= accuracy_score(predictions, valid_targets_fold)
              k=k+1

              #print("Fold ")
              #print(k)
              #print(accuracy_score(predictions, valid_targets_fold))


        print(np.average(cross_val_accuracy))
        k_values_class[i]=np.average(cross_val_accuracy)
    print(k_values_class)

    return

def SGDClassifier():
    labeled_images_data = spio.loadmat("labeled_images.mat")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))
    train_data, test_data, train_targets, test_targets, train_ident, target_ident = splitSet(faces, labels, identities, 0.3)
    runs = np.zeros(10)
    for i in range(len(runs)):
        model = SGDClassifier(loss="log", penalty="l2", n_iter=500, learning_rate='optimal')
        model.fit(train_data, train_targets)
        #predict = model.predict(test_data)
        score = model.score(test_data, test_targets)
        runs[i]=score
        if score > 0.7:
            good_model = model
        print(score)
    print(runs)
    print(good_model.score(test_data, test_targets))

    return

def SVM():
    labeled_images_data = spio.loadmat("labeled_images.mat")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))
    train_data, test_data, train_targets, test_targets, train_ident, target_ident = splitSet(faces, labels, identities, 0.3)
    runs = np.zeros(10)
    for i in range(len(runs)):
        model = svm.LinearSVC()
        model.fit(train_data, train_targets)
        #predict = model.predict(test_data)
        score = model.score(test_data, test_targets)
        runs[i]=score
        if score > 0.7:
            good_model = model
        print(score)
    print(runs)
    print(good_model.score(test_data, test_targets))

    return
def splitSet(data, targets, identities, validRatio):
    "Takes a set of data and returns a validation set and training set"
    number = np.around(len(data)*validRatio)
    borderIdent1 = identities[number]
    borderIdent2 = identities[number+1]
    if borderIdent1!=borderIdent2:
        print("Good split")
    elif borderIdent1==-1 or borderIdent2==-1:
        print("Good split because one or both do not have an identity")
    else:
        print("Bad split")
    train_data = data[number:]
    test_data = data[:number]
    train_ident = identities[number:]
    test_ident =identities[:number]
    train_targ =  targets[number:].squeeze()
    test_targ = targets[:number].squeeze()

    #cross_validation.train_test_split(data, targets, test_size=0.4,random_state=0)
    return train_data, test_data, train_targ, test_targ, train_ident, test_ident


if __name__ == "__main__":
    #knn()
    SVM()
