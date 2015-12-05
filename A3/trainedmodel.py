import scipy.io as spio
import numpy as np
from skimage import io
from time import *
from sklearn import cross_validation, datasets, svm, grid_search
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit, LabelKFold, train_test_split
from sklearn.decomposition import PCA, RandomizedPCA
import pickle
from sklearn.linear_model import SGDClassifier
from pylab import *
from sklearn import preprocessing
from skimage.feature import hog
import cv2
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
SZ=20
bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def save_object(obj, filename):

        pickle.dump(obj, open(filename, "wb"))

def load_object(filename):
    return pickle.load(open(filename, "rb"))

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

def sgdc():
    labeled_images_data = spio.loadmat("labeled_images.mat")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))
    train_data, test_data, train_targets, test_targets, train_ident, target_ident = splitSet(faces, labels, identities, 0.2)
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



def SVM(submit):
    labeled_images_data = spio.loadmat("labeled_images.mat")
    unlabeled_images_data = spio.loadmat("unlabeled_images.mat")
    public_test_data = spio.loadmat("public_test_images.mat")
    faces_test = public_test_data.get("public_test_images")
    unlabeled_faces = unlabeled_images_data.get("unlabeled_images")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    # faces = hog(faces)

    faces = faces.reshape((faces.shape[0], -1))

    unlabeled_faces = unlabeled_faces.transpose(2, 0, 1)
    unlabeled_faces = unlabeled_faces.reshape((unlabeled_faces.shape[0], -1))
    faces_test = faces_test.transpose(2, 0, 1)
    faces_test = faces_test.reshape((faces_test.shape[0], -1))
    #train_data, test_data, train_targets, test_targets, train_ident, target_ident = splitSet(faces, labels, identities, 0.2)
    labels_s = labels.squeeze()

    #train_data, test_data, train_targets, test_targets, train_ident, test_ident = train_test_split(faces, labels_s, identities, train_size=0.9)
    #test = np.intersect1d(train_ident, test_ident)

    # small_faces = faces
    # small_identities = identities
    # small_labels = labels_s
    # aug = np.column_stack((small_identities, small_labels,small_faces))
    #
    # one_array = np.array(filter(lambda row: row[1]==1, aug))
    # two_array = np.array(filter(lambda row: row[1]==2, aug))
    # three_array = np.array(filter(lambda row: row[1]==3, aug))
    # four_array = np.array(filter(lambda row: row[1]==4, aug))
    # five_array = np.array(filter(lambda row: row[1]==5, aug))
    # six_array = np.array(filter(lambda row: row[1]==6, aug))
    # seven_array = np.array(filter(lambda row: row[1]==7, aug))
    #
    # label_arrays = [one_array, two_array, three_array, four_array, five_array, six_array, seven_array]
    #
    # for j in range(len(label_arrays)):
    #     label_arrays[j] = label_arrays[j][label_arrays[j][:,0].argsort()[::-1]]
    #
    #
    # master_array = aug.copy()
    #
    # #save_object(label_arrays, "label_arrays")
    # # label_arrays = load_object("label_arrays")
    #
    # i = 0
    # while i < len(faces):
    #     for j in range(len(label_arrays)):
    #         if i < len(faces) and len(label_arrays[j]>0):
    #             master_array[i] = label_arrays[j][0]
    #             label_arrays[j] = np.delete(label_arrays[j] , 0, axis=0)
    #             #label_arrays[j] = np.zeros(3)
    #             i = i+1
    # # save_object(master_array, "master_canny_100-201")
    # #
    master_array = load_object("master")

    master_ident = master_array[:,0]
    master_array = np.delete(master_array,0,1)
    master_labels = master_array[:,0]
    master_array = np.delete(master_array,0,1)
    master_faces = master_array

    #train_data, test_data, train_targets, test_targets, train_ident, test_ident = splitSet(master_faces, master_labels, master_ident, 0.1)
    #train_data, test_data, train_targets, test_targets, train_ident, test_ident = splitSet(faces, labels_s, identities, 0.3)

    #common_idents_array = np.intersect1d(train_ident, test_ident)


    n_eigenfaces = 121

    # print("-   Performing PCA reduction    -")
    # pca = RandomizedPCA(n_components=n_eigenfaces, whiten=True).fit(unlabeled_faces)
    # save_object(pca, "pca")
    pca = load_object("pca")
    # #train_data = pca.transform(train_data)
    # #test_data = pca.transform(test_data)
    # print("-   Finished PCA reduction    -")
    #
    # print('PCA captures {:.2f} percent of the variance in the dataset'.format(pca.explained_variance_ratio_.sum() * 100))

    tuples = kfold(master_faces,master_labels,master_ident, 9)
    success_rates_train = []
    success_rate_valid = []
    if not submit:
        for tuple in tuples:
            train_data, test_data, train_targets, test_targets, train_ident, test_ident= tuple
            # train_data = pca.transform(train_data)
            # test_data = pca.transform(test_data)
            classifier = svm.SVC( gamma=0.5, C=1, kernel='linear')
            model = BaggingClassifier(classifier, n_estimators=10, bootstrap=True, verbose=1)
            model.fit(train_data, train_targets)


            #Train
            score = model.score(train_data, train_targets)
            valid_score = model.score(test_data, test_targets)

            print("Training :")
            print(score)
            success_rates_train.append(score)

            #Validation
            print("Validation :")
            print(valid_score)
            success_rate_valid.append(valid_score)

        print("Training rates :")
        print(success_rates_train)
        print("Training average :")
        print(np.average(success_rates_train))

        print("Validation rates :")
        print(success_rate_valid)
        print("Validation average :")
        print(np.average(success_rate_valid))
    if submit:
        train_data, test_data, train_targets, test_targets, train_ident, test_ident = splitSet(master_faces, master_labels, master_ident, 0.2)
        classification = svm.SVC( gamma=0.5, C=1, kernel='poly')
        model = BaggingClassifier(classification, n_estimators=20, max_features=10, bootstrap=True, verbose=1)
        model.fit(master_faces, master_labels)
        test_predictions = model.predict(faces_test)


        #Test predictions

        ascending = np.zeros(1253)

        for i in range(len(ascending)):
             ascending[i]=i+1
        ascending = ascending.astype(int)
        hidden_guesses = -1*np.ones(1253-len(test_predictions))
        test_predictions = np.concatenate([test_predictions, hidden_guesses])
        test_predictions = test_predictions.astype(int)
        csv = np.column_stack((ascending, test_predictions))
        np.savetxt("test_faces_est20_max10.csv", csv, delimiter=",")
    return



def med_blur(data):
    for i in range(len(data)):
        data[i] = cv2.medianBlur(data[i],5)
    return data

def splitSet(data, targets, identities, validRatio):
    "Takes a set of data and returns a validation set and training set"
    number = np.around(len(data)*validRatio)
    borderIdent1 = identities[number]
    borderIdent2 = identities[number+1]
    #if borderIdent1!=borderIdent2:
    #    print("Good split")
    #elif borderIdent1==-1 or borderIdent2==-1:
    #    print("Good split because one or both do not have an identity")
    #else:
    #    print("Bad split")
    train_data = data[number:]
    test_data = data[:number]
    train_ident = identities[number:]
    test_ident =identities[:number]
    train_targ =  targets[number:].squeeze()
    test_targ = targets[:number].squeeze()

    #cross_validation.train_test_split(data, targets, test_size=0.4,random_state=0)
    return train_data, test_data, train_targ, test_targ, train_ident, test_ident

def kfold(data, targets, identities, folds):
    data_split = np.split(data, folds)
    targets_split = np.split(targets, folds)
    identities_split = np.split(identities, folds)

    tuples = []
    for i in range(folds):
        valid_data = data_split[i]
        valid_targets = targets_split[i]
        valid_ident = identities_split[i]
        train_data = np.concatenate(tuple(map(tuple, np.delete(data_split,i, 0))))
        train_targets = np.concatenate(tuple(map(tuple, np.delete(targets_split,i, 0))))
        train_ident = np.concatenate(tuple(map(tuple, np.delete(identities_split,i, 0))))
        tuples.append((train_data,valid_data , train_targets, valid_targets, train_ident, valid_ident))
    return tuples

if __name__ == "__main__":
    #knn()
    #semi_supervised()
    SVM(True)
