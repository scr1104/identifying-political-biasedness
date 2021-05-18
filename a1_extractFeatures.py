#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import os
import csv

bristol_dict = {}
warriner_dict = {}
center_feats = []
right_feats = []
left_feats = []
alt_feats = []
center_id = []
right_id = []
left_id = []
alt_id = []


# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
PUNCT_TAG = {
    '#', '$', '.', ',', ':', '"', '‘', "’", '“', '”', "-LRB-", "-RRB-", "HYPH", "NFP"}


def bristol(args):
    """ This helper function opens "BristolNorms+GilhoolyLogie.csv" and process to return it as a dictionary

        Parameters:
            args : parsed argument
    """
    global bristol_dict

    bristol_dir = os.path.join(args.a1_dir, '../Wordlists/BristolNorms+GilhoolyLogie.csv')
    print(bristol_dir)


    with open(bristol_dir, 'r') as rf:
        reader = csv.reader(rf, delimiter=',')
        for row in reader:
            try:
                AoA = float(row[3])
            except ValueError:
                AoA = 0
            try:
                IMG = float(row[4])
            except ValueError:
                IMG = 0
            try:
                FAM = float(row[5])
            except ValueError:
                FAM = 0

            bristol_dict[row[1]] = [AoA, IMG, FAM]


def warriner(args):
    """ This helper function opens "Ratings_Warriner_et_al.csv" and process to return it as a dictionary

        Parameters:
            args : parsed argument
    """
    global warriner_dict

    warriner_dir = os.path.join(args.a1_dir, '../Wordlists/Ratings_Warriner_et_al.csv')

    with open(warriner_dir, 'r') as rf:
        reader = csv.reader(rf, delimiter=',')
        for row in reader:
            try:
                VMS = float(row[2])
            except ValueError:
                VMS = 0
            try:
                AMS = float(row[5])
            except ValueError:
                AMS = 0
            try:
                DMS = float(row[8])
            except ValueError:
                DMS = 0
            warriner_dict[row[1]] = [VMS, AMS, DMS]


def LIWC(args):
    """ This helper function opens "_feats.dat.npy" files and save each of them as arrays of arrays.

        Parameters:
            args : parsed argument
    """
    global center_feats, left_feats, right_feats, alt_feats

    center_dir = os.path.join(args.a1_dir, 'feats/Center_feats.dat.npy')
    right_dir = os.path.join(args.a1_dir, 'feats/Right_feats.dat.npy')
    left_dir = os.path.join(args.a1_dir, 'feats/Left_feats.dat.npy')
    alt_dir = os.path.join(args.a1_dir, 'feats/Alt_feats.dat.npy')

    center_feats = np.load(center_dir)
    right_feats = np.load(right_dir)
    left_feats = np.load(left_dir)
    alt_feats = np.load(alt_dir)


def IDs(args):
    """ This helper function opens "_IDs.txt" files and save each of them as a list.

        Parameters:
            args : parsed argument
    """
    global center_id, left_id, right_id, alt_id

    center_dir = os.path.join(args.a1_dir + 'feats/Center_IDs.txt')
    right_dir = os.path.join(args.a1_dir + 'feats/Right_IDs.txt')
    left_dir = os.path.join(args.a1_dir + 'feats/Left_IDs.txt')
    alt_dir = os.path.join(args.a1_dir + 'feats/Alt_IDs.txt')

    center = open(center_dir, "r")
    right = open(right_dir, "r")
    left = open(left_dir, "r")
    alt = open(alt_dir, "r")

    center_id = center.read().splitlines()
    right_id = right.read().splitlines()
    left_id = left.read().splitlines()
    alt_id = alt.read().splitlines()

    center.close()
    right.close()
    left.close()
    alt.close()


def extract1(comment):
    """ This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    """

    # Extract features that rely on capitalization.
    feat_extract1 = np.zeros(173)
    if len(comment) == 0:
        return feat_extract1
    sentence_list = comment.split("\n")[:-1]

    # 17
    feat_extract1[16] = len(sentence_list)

    total_token_counts = 0
    total_token_counts_char = 0
    tokens_list = []
    word_list = []
    pos_list = []
    for s in sentence_list:
        sentence = s.split(" ")
        tokens_list.append(sentence)

    # 15
    for i in tokens_list:
        total_token_counts += len(i)
    feat_extract1[14] = total_token_counts / len(tokens_list)

    for i in tokens_list:
        words = []
        poss = []
        for j in i:
            index = j.rfind("/")
            words.append(j[:index])
            poss.append(j[index + 1:])
        word_list.append(words)
        pos_list.append(poss)

    lower_word_list = []

    for s in word_list:
        # 1
        feat_extract1[0] += sum(map(lambda x: (x.isupper()) and (len(x) >= 3), s))

        # 2, 3, 4
        feat_extract1[1] += sum(map(lambda x: x.lower() in FIRST_PERSON_PRONOUNS, s))
        feat_extract1[2] += sum(map(lambda x: x.lower() in SECOND_PERSON_PRONOUNS, s))
        feat_extract1[3] += sum(map(lambda x: x.lower() in THIRD_PERSON_PRONOUNS, s))

        # 8
        feat_extract1[7] += s.count(",")

        # 14
        feat_extract1[13] += sum(map(lambda x: x.lower() in SLANG, s))

        # Lowercase the text in comment for later.
        lower_s = [word.lower() for word in s]
        lower_word_list += lower_s

    for s in pos_list:
        # 5, 6
        feat_extract1[4] += s.count("CC")
        feat_extract1[5] += s.count("VBD")

        # 10, 11, 12, 13
        feat_extract1[9] += sum(map(lambda x: x in ["NN", "NNS"], s))
        feat_extract1[10] += sum(map(lambda x: x in ["NNP", "NNPS"], s))
        feat_extract1[11] += sum(map(lambda x: x in ["RB", "RBR", "RBS"], s))
        feat_extract1[12] += sum(map(lambda x: x in ["WDT", "WP", "WP$", "WRB"], s))

        # 16
        total_token_counts_char += sum(map(lambda x: x not in PUNCT_TAG, s))
    # 16
    feat_extract1[15] = total_token_counts_char / len(pos_list)

    future_num = 0
    for s in tokens_list:
        # 7
        lowered = [x.lower() for x in s]
        future_num += lowered.count("will/md")

    for i in range(len(tokens_list)):
        for j in range(len(tokens_list[i])):
            # 9
            if (pos_list[i][j] in PUNCT_TAG) and (len(word_list[i][j]) > 1):
                feat_extract1[8] += 1
            # 7
            try:
                if (tokens_list[i][j].lower() == "go/vbg") & (pos_list[i][j + 1] == "TO") & \
                        (pos_list[i][j + 2] == "VB"):
                    future_num += 1
            except IndexError:
                pass

    feat_extract1[6] = future_num

    AoA = []
    IMG = []
    FAM = []
    VMS = []
    AMS = []
    DMS = []

    for word in lower_word_list:
        if word in bristol_dict:
            AoA.append(bristol_dict[word][0])
            IMG.append(bristol_dict[word][1])
            FAM.append(bristol_dict[word][2])
        if word in warriner_dict:
            VMS.append(warriner_dict[word][0])
            AMS.append(warriner_dict[word][1])
            DMS.append(warriner_dict[word][2])

    for feat in [AoA, IMG, FAM, VMS, AMS, DMS]:
        if len(feat) == 0:
            feat.append(0)

    # 18, 19, 20
    if len(AoA) > 0:
        feat_extract1[17] = np.mean(AoA)
        feat_extract1[18] = np.mean(IMG)
        feat_extract1[19] = np.mean(FAM)
    # 21, 22, 23
        feat_extract1[20] = np.std(AoA)
        feat_extract1[21] = np.std(IMG)
        feat_extract1[22] = np.std(FAM)

    # 24, 25, 26
    if len(VMS) > 0:
        feat_extract1[23] = np.mean(VMS)
        feat_extract1[24] = np.mean(AMS)
        feat_extract1[25] = np.mean(DMS)
    # 27, 28, 29
        feat_extract1[26] = np.std(VMS)
        feat_extract1[27] = np.std(AMS)
        feat_extract1[28] = np.std(DMS)

    return feat_extract1


def extract2(feat, comment_class, comment_id):
    """ This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    """
    if comment_class == "Center":
        index = center_id.index(comment_id)
        liwc = center_feats[index]
        for i in range(29, 173):
            feat[i] = liwc[i-29]
    elif comment_class == "Right":
        index = right_id.index(comment_id)
        liwc = right_feats[index]
        for i in range(29, 173):
            feat[i] = liwc[i-29]
    elif comment_class == "Left":
        index = left_id.index(comment_id)
        liwc = left_feats[index]
        for i in range(29, 173):
            feat[i] = liwc[i-29]
    elif comment_class == "Alt":
        index = alt_id.index(comment_id)
        liwc = alt_feats[index]
        for i in range(29, 173):
            feat[i] = liwc[i-29]
    return feat


def main(args):
    # Declare necessary global variables here.
    cat_dic = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    bristol(args)
    warriner(args)
    LIWC(args)
    IDs(args)

    for i in range(len(data)):
        full = data[i]
        id = full["id"]
        body = full["body"]
        cat = full["cat"]

        feat = extract1(body)
        feat = extract2(feat, cat, id)
        feats[i] = np.concatenate([feat, np.array([cat_dic[cat]])])

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                        default='/u/cs401/A1/')
    args = parser.parse_args()

    main(args)

