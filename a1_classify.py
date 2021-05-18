#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from random import sample

# set the random state for reproducibility
import numpy as np

np.random.seed(401)


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    all_classification = 0
    correct_classification = 0
    for i in range(len(C)):
        for j in range(len(C[i])):
            all_classification += C[i][j]
            if i == j:
                correct_classification += C[i][j]

    return correct_classification / all_classification


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall_list = []
    for k in range(len(C)):
        recall_list.append(C[k][k] / sum(C[k]))

    return recall_list


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision_list = []

    for i in range(len(C[0])):
        denom = 0
        for k in range(len(C)):
            denom += C[k][i]
        precision_list.append(C[i][i] / denom)

    return precision_list


def xy_split():
    global X_train, Y_train, X_test, Y_test
    l1 = len(train_data)
    x_train = np.empty(l1, dtype=np.ndarray)
    y_train = np.zeros(l1, dtype=int) - 1
    for i in range(l1):
        sl = np.split(train_data[i], [173])
        x_train[i] = sl[0]
        y_train[i] = sl[1][0]
    X_train = x_train
    Y_train = y_train
    l2 = len(test_data)
    x_test = np.empty(l2, dtype=np.ndarray)
    y_test = np.zeros(l2, dtype=int) - 1
    for i in range(l2):
        sl = np.split(test_data[i], [173])
        x_test[i] = sl[0]
        y_test[i] = sl[1][0]
    X_test = x_test
    Y_test = y_test


def class31(output_dir, X_train, X_test, y_train, y_test):
    """ This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    """
    print("3.1 processing...")

    x_train = list(X_train)
    x_test = list(X_test)
    clf1 = SGDClassifier()
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier(max_depth=5, n_estimators=10)
    clf4 = MLPClassifier(alpha=0.05)
    clf5 = AdaBoostClassifier()

    clf1.fit(x_train, y_train)
    clf2.fit(x_train, y_train)
    clf3.fit(x_train, y_train)
    clf4.fit(x_train, Y_train)
    clf5.fit(x_train, y_train)

    pred1 = clf1.predict(x_test)
    pred2 = clf2.predict(x_test)
    pred3 = clf3.predict(x_test)
    pred4 = clf4.predict(x_test)
    pred5 = clf5.predict(x_test)

    pred_list = [pred1, pred2, pred3, pred4, pred5]

    classifier_name = ["SGDClassifier", "GaussianNB", "RandomForestClassifier", "MLPClassifier", "AdaBoostClassifier"]

    accuracy_list = []

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:

        for i in range(len(pred_list)):
            conf_matrix = confusion_matrix(y_test, pred_list[i])
            acc = accuracy(conf_matrix)
            accuracy_list.append(acc)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)
            outf.write(f'Results for {classifier_name[i]}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    i = accuracy_list.index(max(accuracy_list))
    return i


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''

    print("3.2 processing...")

    x_test = list(X_test)
    num_train = [1000, 5000, 10000, 15000, 20000]
    classifiers = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05), AdaBoostClassifier()]

    index_1000 = np.random.randint(0, 32000, size=1000)
    X_1k = [X_train[i] for i in index_1000]
    y_1k = [y_train[i] for i in index_1000]

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        for num in num_train:
            X_train_chosen, y_train_chosen = zip(*sample(list(zip(X_train, y_train)), num))
            clf = classifiers[iBest]
            clf.fit(X_train_chosen, y_train_chosen)
            prediction = clf.predict(x_test)
            C = confusion_matrix(y_test, prediction)
            acc = accuracy(C)
            outf.write(f'{num}: {acc:.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    print("3.3 processing...")

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # 1
        x_train = list(X_train)
        selector_k5 = SelectKBest(f_classif, k=5)
        selector_k5_1k = SelectKBest(f_classif, k=5)
        selector_k50 = SelectKBest(f_classif, k=50)
        selector_k50_1k = SelectKBest(f_classif, k=50)
        X_train_k5_full = selector_k5.fit_transform(x_train, y_train)
        X_train_k5_1k = selector_k5_1k.fit_transform(X_1k, y_1k)
        selector_k50.fit_transform(x_train, y_train)
        selector_k50_1k.fit_transform(X_1k, y_1k)

        pp_k5 = selector_k5.pvalues_
        pp_k50 = selector_k50.pvalues_

        k50_full_index = list(selector_k50.get_support(indices=True))
        k5_full_index = list(selector_k5.get_support(indices=True))
        k5_1k_index = list(selector_k5_1k.get_support(indices=True))

        outf.write(f'{5} p-values: {[format(pp_k5[i]) for i in k5_full_index]}\n')
        outf.write(f'{50} p-values: {[format(pp_k50[i]) for i in k50_full_index]}\n')

        # 2
        classifiers1 = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10),
                        MLPClassifier(alpha=0.05), AdaBoostClassifier()]
        classifiers2 = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10),
                        MLPClassifier(alpha=0.05), AdaBoostClassifier()]

        classifier_1k = classifiers1[i]
        classifier_full = classifiers2[i]

        classifier_1k.fit(X_train_k5_1k, y_1k)
        classifier_full.fit(X_train_k5_full, y_train)

        X_test_reduced_full = np.zeros(len(X_test), dtype=np.object)
        X_test_reduced_1k = np.zeros(len(X_1k), dtype=np.object)

        for i in range(len(X_test)):
            feats = np.zeros(5)
            for j in range(5):
                feats[j] = X_test[i][k5_full_index[j]]
            X_test_reduced_full[i] = feats

        for i in range(len(X_1k)):
            feats = np.zeros(5)
            for j in range(5):
                feats[j] = X_1k[i][k5_full_index[j]]
            X_test_reduced_1k[i] = feats

        pred_1k = classifier_1k.predict(list(X_test_reduced_1k))
        pred_full = classifier_full.predict(list(X_test_reduced_full))

        C_1k = confusion_matrix(pred_1k, y_1k)
        C_full = confusion_matrix(pred_full, y_test)

        acc_1k = accuracy(C_1k)
        acc_full = accuracy(C_full)

        outf.write(f'Accuracy for 1k: {acc_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {acc_full:.4f}\n')

        # 3
        feat_intersection = set([value for value in k5_full_index if value in k5_1k_index])

        outf.write(f'Chosen feature intersection: {feat_intersection}\n')

        # 4
        outf.write(f'Top-5 at higher: {set(k5_full_index)}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
    '''

    print("3.4 processing...")

    all_x = np.concatenate((X_train, X_test))
    all_y = np.concatenate((y_train, y_test))
    kf5 = KFold(shuffle=True)
    acc_clf_list = [[], [], [], [], []]

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        for train_index, test_index in kf5.split(all_x, all_y):
            print("fold begins:")
            fold_train_X, fold_test_X = list(all_x[train_index]), list(all_x[test_index])
            fold_train_y, fold_test_y = all_y[train_index], all_y[test_index]

            clf1 = SGDClassifier()
            clf2 = GaussianNB()
            clf3 = RandomForestClassifier(max_depth=5, n_estimators=10)
            clf4 = MLPClassifier(alpha=0.05)
            clf5 = AdaBoostClassifier()

            clf1.fit(fold_train_X, fold_train_y)
            clf2.fit(fold_train_X, fold_train_y)
            clf3.fit(fold_train_X, fold_train_y)
            clf4.fit(fold_train_X, fold_train_y)
            clf5.fit(fold_train_X, fold_train_y)

            pred1 = clf1.predict(fold_test_X)
            pred2 = clf2.predict(fold_test_X)
            pred3 = clf3.predict(fold_test_X)
            pred4 = clf4.predict(fold_test_X)
            pred5 = clf5.predict(fold_test_X)

            C1 = confusion_matrix(fold_test_y, pred1)
            C2 = confusion_matrix(fold_test_y, pred2)
            C3 = confusion_matrix(fold_test_y, pred3)
            C4 = confusion_matrix(fold_test_y, pred4)
            C5 = confusion_matrix(fold_test_y, pred5)

            acc1 = accuracy(C1)
            acc2 = accuracy(C2)
            acc3 = accuracy(C3)
            acc4 = accuracy(C4)
            acc5 = accuracy(C5)

            acc_list = [acc1, acc2, acc3, acc4, acc5]

            acc_clf_list[0].append(acc1)
            acc_clf_list[1].append(acc2)
            acc_clf_list[2].append(acc3)
            acc_clf_list[3].append(acc4)
            acc_clf_list[4].append(acc5)

            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in acc_list]}\n')

        p_values_1 = ttest_rel(acc_clf_list[0], acc_clf_list[4]).pvalue
        p_values_2 = ttest_rel(acc_clf_list[1], acc_clf_list[4]).pvalue
        p_values_3 = ttest_rel(acc_clf_list[2], acc_clf_list[4]).pvalue
        p_values_4 = ttest_rel(acc_clf_list[3], acc_clf_list[4]).pvalue
        p_values = [p_values_1, p_values_2, p_values_3, p_values_4]

        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # load data and split into train(80%) and test(20%).
    X_train = 0
    Y_train = 0
    X_test = 0
    Y_test = 0
    f = np.load(args.input, "r")
    keys_list = list(f)
    input_data = f[keys_list[0]]
    split_data = train_test_split(input_data, test_size=0.2, train_size=0.8)
    train_data = split_data[0]
    test_data = split_data[1]
    xy_split()
    iBest = class31(args.output_dir, X_train, X_test, Y_train, Y_test)
    (X_1k, y_1k) = class32(args.output_dir, X_train, X_test, Y_train, Y_test, iBest)
    class33(args.output_dir, X_train, X_test, Y_train, Y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, Y_train, Y_test, iBest)

