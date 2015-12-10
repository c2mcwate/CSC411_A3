import scipy.io as spio
import numpy as np
from skimage import data, exposure, img_as_float, feature
from time import *
import scipy
import scipy.ndimage
import numpy.ma as ma
from skimage import exposure
from skimage.filters import rank
from sklearn import cross_validation, datasets, svm, grid_search
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit, LabelKFold, train_test_split
from sklearn.decomposition import PCA, RandomizedPCA
import pickle
from sklearn.linear_model import SGDClassifier
from pylab import *
from skimage.feature import hog
#import cv2
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn import preprocessing
from skimage.filters.rank import *
#from cv2 import GaussianBlur
from skimage.morphology import disk
from skimage.exposure import *
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import cv2
from skimage.color import rgb2gray
from sklearn import mixture
from sklearn import linear_model

def save_object(obj, filename):

        pickle.dump(obj, open(filename, "wb"))

def load_object(filename):
    return pickle.load(open(filename, "rb"))

def SVM(submit):
    labeled_images_data = spio.loadmat("labeled_images.mat")
    unlabeled_images_data = spio.loadmat("unlabeled_images.mat")
    public_test_data = spio.loadmat("public_test_images.mat")
    hidden_test_data = spio.loadmat("hidden_test_images.mat")
    hidden_faces = hidden_test_data.get("hidden_test_images")
    faces_test = public_test_data.get("public_test_images")
    unlabeled_faces = unlabeled_images_data.get("unlabeled_images")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))
    hidden_faces = hidden_faces.transpose(2, 0, 1)
    hidden_faces = hidden_faces.reshape((hidden_faces.shape[0], -1))

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
    #             if(j==6):
    #                  master_array[i] = label_arrays[j][0]
    #                  label_arrays[j] = np.delete(label_arrays[j] , 0, axis=0)
    #                  i = i+1
    #             master_array[i] = label_arrays[j][0]
    #             label_arrays[j] = np.delete(label_arrays[j] , 0, axis=0)
    #             #label_arrays[j] = np.zeros(3)
    #             i = i+1
    # #save_object(master_array, "master_canny_100-201")

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
    # pca = load_object("pca")
    # #train_data = pca.transform(train_data)
    # #test_data = pca.transform(test_data)
    # print("-   Finished PCA reduction    -")
    #
    # print('PCA captures {:.2f} percent of the variance in the dataset'.format(pca.explained_variance_ratio_.sum() * 100))


    # PUT YOUR PROCESSING HERE
    #Reshape
    hidden_faces= preprocessing.normalize(hidden_faces, norm='l2')
    master_faces = preprocessing.normalize(master_faces, norm='l2')
    faces_test = preprocessing.normalize(faces_test, norm='l2')
    hidden_faces = hidden_faces.reshape(len(hidden_faces), 32, 32)

    master_faces = master_faces.reshape(len(master_faces), 32, 32)
    faces_test = faces_test.reshape(len(faces_test), 32, 32)
    plt.subplot(122),plt.imshow(faces_test[3], cmap='gray')
    plt.title('Normal'), plt.xticks([]), plt.yticks([])
    # plt.show()



    #Gamma correction
    hidden_faces = all_gamma(hidden_faces)

    master_faces = all_gamma(master_faces)
    faces_test = all_gamma(faces_test)
    plt.subplot(122),plt.imshow(faces_test[3], cmap='gray')
    plt.title('Gamma correction'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # #Dog filter
    # master_faces -= cv2.GaussianBlur(master_faces, (3, 3),1)
    # faces_test -= cv2.GaussianBlur(faces_test, (3, 3),1)
    # plt.subplot(122),plt.imshow(faces_test[1], cmap='gray')
    # plt.title('Dog Filter'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # #Rescale intensity
    # master_faces = testing(master_faces)
    # faces_test = testing(faces_test)
    #
    # plt.subplot(122),plt.imshow(faces_test[15], cmap='gray')
    # plt.title('Rescale'), plt.xticks([]), plt.yticks([])
    # plt.show()

    #Equalization of variance TODO
    hidden_faces = EQ(hidden_faces)
    master_faces = EQ(master_faces)

    faces_test = EQ(faces_test)
    plt.subplot(122),plt.imshow(faces_test[3], cmap='gray')
    plt.title('Equalization'), plt.xticks([]), plt.yticks([])
    # plt.show()

    #Reshape
    master_faces = master_faces.reshape((master_faces.shape[0], -1))
    faces_test = faces_test.reshape((faces_test.shape[0], -1))
    hidden_faces = hidden_faces.reshape((hidden_faces.shape[0], -1))

    tuples = kfold(master_faces,master_labels,master_ident, 13)
    success_rates_train = []
    success_rate_valid = []
    if not submit:
        for tuple in tuples:

            train_data, test_data, train_targets, test_targets, train_ident, test_ident= tuple
            # train_data = pca.transform(train_data)
            # test_data = pca.transform(test_data)

            classifier = svm.SVC( gamma=0.5, C=1, kernel='poly')
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
        classification = svm.SVC( gamma=0.5, C=1, kernel='poly')
        model = BaggingClassifier(classification, n_estimators=20, bootstrap_features=True, bootstrap=True, verbose=1)
        model.fit(master_faces, master_labels)
        test_predictions = model.predict(faces_test)
        hidden_predictions = model.predict(hidden_faces)


        #Test predictions

        ascending = np.zeros(1253)

        for i in range(len(ascending)):
             ascending[i]=i+1
        ascending = ascending.astype(int)
        hidden_guesses = hidden_predictions
        test_predictions = np.concatenate([test_predictions, hidden_guesses])
        test_predictions = test_predictions.astype(int)
        csv = np.column_stack((ascending, test_predictions))
        np.savetxt("hidden.csv", csv, delimiter=",")
    return

def testing(array):
    for i in range(len(array)):
        array[i] = exposure.rescale_intensity(array[i])
    return array

def all_gamma(array):
    for i in range(len(array)):
        array[i] = exposure.adjust_gamma(array[i], 0.2)
    return array

def EQ(array):
    for i in range(len(array)):
        array[i] = equalize(array[i], disk(20))
    return array

def adapt(array):
    for i in range(len(array)):
        p2, p98 = np.percentile(array[i], (2, 98))
        array[i] = exposure.rescale_intensity(array[i], in_range=(p2, p98))

    return array

def KNN(submit):
    labeled_images_data = spio.loadmat("labeled_images.mat")
    unlabeled_images_data = spio.loadmat("unlabeled_images.mat")
    public_test_data = spio.loadmat("public_test_images.mat")
    faces_test = public_test_data.get("hidden_images")
    unlabeled_faces = unlabeled_images_data.get("unlabeled_images")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))
    unlabeled_faces = unlabeled_faces.transpose(2, 0, 1)
    unlabeled_faces = unlabeled_faces.reshape((unlabeled_faces.shape[0], -1))
    faces_test = faces_test.transpose(2, 0, 1)
    faces_test = faces_test.reshape((faces_test.shape[0], -1))
    labels_s = labels.squeeze()


    master_array = load_object("master")

    master_ident = master_array[:,0]
    master_array = np.delete(master_array,0,1)
    master_labels = master_array[:,0]
    master_array = np.delete(master_array,0,1)
    master_faces = master_array



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
    #Reshape
    master_faces = preprocessing.normalize(master_faces, norm='l2')
    faces_test = preprocessing.normalize(faces_test, norm='l2')
    master_faces = master_faces.reshape(len(master_faces), 32, 32)
    faces_test = faces_test.reshape(len(faces_test), 32, 32)
    plt.subplot(122),plt.imshow(faces_test[3], cmap='gray')
    plt.title('Normal'), plt.xticks([]), plt.yticks([])
    plt.show()

    #Gamma correction
    master_faces = all_gamma(master_faces)
    faces_test = all_gamma(faces_test)
    plt.subplot(122),plt.imshow(faces_test[3], cmap='gray')
    plt.title('Gamma correction'), plt.xticks([]), plt.yticks([])
    plt.show()


    #Equalization of variance TODO
    master_faces = EQ(master_faces)
    faces_test = EQ(faces_test)
    plt.subplot(122),plt.imshow(faces_test[3], cmap='gray')
    plt.title('Equalization'), plt.xticks([]), plt.yticks([])
    plt.show()

    #Reshape
    master_faces = master_faces.reshape((master_faces.shape[0], -1))
    faces_test = faces_test.reshape((faces_test.shape[0], -1))

    tuples = kfold(master_faces,master_labels,master_ident, 13)
    success_rates_train = []
    success_rate_valid = []
    if not submit:
        for tuple in tuples:
            train_data, test_data, train_targets, test_targets, train_ident, test_ident= tuple
            #train_data = pca.transform(train_data)
            #test_data = pca.transform(test_data)
            model = KNeighborsClassifier(n_neighbors=21, weights='distance', algorithm='auto')
            #model = BaggingClassifier(classifier, n_estimators=10, bootstrap=True, verbose=1)
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
        model = BaggingClassifier(classification, n_estimators=30, bootstrap=True, verbose=1)
        model.fit(master_faces, master_labels)
        test_predictions = model.predict(faces_test)


        #Test predictions

        ascending = np.zeros(1253)

        for i in range(len(ascending)):
             ascending[i]=i+1
        ascending = ascending.astype(int)
        test_predictions = test_predictions.astype(int)
        csv = np.column_stack((ascending, test_predictions))
        np.savetxt("knn.csv", csv, delimiter=",")
    return

def Logit(submit):
    labeled_images_data = spio.loadmat("labeled_images.mat")
    unlabeled_images_data = spio.loadmat("unlabeled_images.mat")
    public_test_data = spio.loadmat("public_test_images.mat")
    faces_test = public_test_data.get("public_test_images")
    unlabeled_faces = unlabeled_images_data.get("unlabeled_images")
    labels = labeled_images_data.get("tr_labels")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)
    faces = faces.reshape((faces.shape[0], -1))
    unlabeled_faces = unlabeled_faces.transpose(2, 0, 1)
    unlabeled_faces = unlabeled_faces.reshape((unlabeled_faces.shape[0], -1))
    faces_test = faces_test.transpose(2, 0, 1)
    faces_test = faces_test.reshape((faces_test.shape[0], -1))
    labels_s = labels.squeeze()


    master_array = load_object("master")

    master_ident = master_array[:,0]
    master_array = np.delete(master_array,0,1)
    master_labels = master_array[:,0]
    master_array = np.delete(master_array,0,1)
    master_faces = master_array




    n_eigenfaces = 121

    print("-   Performing PCA reduction    -")
    pca = RandomizedPCA(n_components=121, whiten=True).fit(unlabeled_faces)
    #save_object(pca, "pca")
    #train_data = pca.transform(train_data)
    #test_data = pca.transform(test_data)
    print("-   Finished PCA reduction    -")

    print('PCA captures {:.2f} percent of the variance in the dataset'.format(pca.explained_variance_ratio_.sum() * 100))

    c_value_success_valid = np.zeros(5)
    c_value_success_train = np.zeros(5)
    c_values = [0.0005, 0.005, 0.05, 0.5, 5]

    for i in range(len(c_values)):

        tuples = kfold(master_faces,master_labels,master_ident, 13)
        success_rates_train = []
        success_rate_valid = []
        if not submit:
            for tuple in tuples:
                train_data, test_data, train_targets, test_targets, train_ident, test_ident= tuple
                # train_data = pca.transform(train_data)
                # test_data = pca.transform(test_data)
                model = linear_model.LogisticRegression(penalty='l2', C=c_values[i], max_iter=10000)
                model.fit(train_data, train_targets)


                #Train
                score = model.score(train_data, train_targets)
                valid_score = model.score(test_data, test_targets)

                # print("Training :")
                # print(score)
                success_rates_train.append(score)

                # #Validation
                # print("Validation :")
                # print(valid_score)
                success_rate_valid.append(valid_score)

            print("Training rates :")
            print(success_rates_train)
            print("Training average :")
            print(np.average(success_rates_train))
            c_value_success_train[i] = np.average(success_rates_train)
            print("Validation rates :")
            print(success_rate_valid)
            print("Validation average :")
            print(np.average(success_rate_valid))
            c_value_success_valid[i] = np.average(success_rate_valid)
    #pd.plot_digits(train_data)
    plot.figure(1)
    plot.plot(c_values, c_value_success_valid, label="Validation", marker="o")
    plot.plot(c_values, c_value_success_train, label="Training", marker="o")
    plot.xlabel("c value")
    plot.ylabel("Classification Success Rate")
    plot.legend()
    #plot.show()
    plot.title('C value vs Success rate ( Logistic Regression)')
    plot.savefig("logit_graph", ext="png", close=False, verbose=True)
    if submit:
        classification = linear_model.LogisticRegression(penalty='l2', C=0.5, max_iter=10000)
        master_faces = pca.transform(master_faces)
        faces_test = pca.transform(faces_test)
        model = BaggingClassifier(classification, n_estimators=20, bootstrap=True, verbose=1)
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
        np.savetxt("logit.csv", csv, delimiter=",")
    return

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
    #KNN(True)
    #semi_supervised()
    #SVM(True)

    Logit(False)