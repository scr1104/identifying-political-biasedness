import numpy as np
import argparse
import json
import os
import csv
import statistics as stat
import re
import time

center_feats = []
right_feats = []
left_feats = []
alt_feats = []
center_id = []
right_id = []
left_id = []
alt_id = []

proc_center = [{'id': 'dh7eoih', 'cat': 'Center',
                 'body': 'so/RB terrible/JJ that/IN it/PRP take/VBD microtargete/VBN fake/JJ news/NN campaign/NNS '
                         'orchestrate/VBN by/IN a/DT hostile/JJ foreign/JJ power/NN to/TO damage/VB her/PRP and/CC '
                         'even/RB '
                         'then/RB she/PRP still/RB win/VBD the/DT popular/JJ vote/NN by/IN million/NNS ./.\n'},
                {'id': 'dh7ezth', 'cat': 'Center',
                 'body': ">/XX as/IN O’Brien/NNP pass/VBD the/DT telescreen/NN a/DT thought/NN seem/VBD to/TO "
                         "strike/VB him/PRP ./.\nHe/PRP stop/VBD ,/, turn/VBD aside/RB and/CC press/VBD a/DT "
                         "switch/NN on/IN the/DT wall/NN ./.\nthere/EX be/VBD a/DT sharp/JJ snap/NN ./.\nthe/DT "
                         "voice/NN have/VBD stop/VBN ./.\n>/XX Julia/NNP utter/VBD a/DT tiny/JJ sound/NN ,/, "
                         "a/DT sort/NN of/IN squeak/NN of/IN surprise/NN ./.\neven/RB in/IN the/DT midst/NN of/IN "
                         "his/PRP$ panic/NN ,/, Winston/NNP be/VBD too/RB much/RB take/VBN aback/RB to/TO be/VB "
                         "able/JJ to/TO hold/VB his/PRP$ tongue/NN ./.\n>/XX '/`` You/PRP can/MD turn/VB it/PRP "
                         "off/RP !/. '/''\nhe/PRP say/VBD ./.\n>/XX '/`` yes/UH ,/, '/'' say/VBD O’Brien/NNP ,/, "
                         "'/`` we/PRP can/MD turn/VB it/PRP off/RP ./.\nWe/PRP have/VBP that/DT privilege/NN ./. '/'' "
                         "-/:\nGeorge/NNP Orwell/NNP ,/, 1984/CD\n"},
                {'id': 'dh7fo27', 'cat': 'Center',
                 'body': 'It/PRP be/VBZ amazing/JJ they/PRP have/VB have/VBN a/DT legitimate/JJ candidate/NN like/IN '
                         'Huntsman/NNP and/CC he/PRP do/VBD not/RB even/RB sniff/VB a/DT chance/NN .../: but/CC '
                         'that/DT be/VBD against/IN Obama/NNP ./.\nI/PRP still/RB do/VBP not/RB think/VB he/PRP '
                         'would/MD have/VB have/VBN a/DT chance/NN against/IN Trump/NNP ./.\n'}]
proc_right = [{'id': 'd25vk7s', 'cat': 'Right', 'body': 'I/PRP love/VBP how/WRB max/NNP only/RB get/VBD 5/CD line/NNS '
                                                        'max/NN ,/, and/CC this/DT gary/NNP dude/NNP get/VBD 10/CD\n'},
              {'id': 'd25vk82', 'cat': 'Right', 'body': 'the/DT fuck/NN ,/, do/VBD some/DT guy/NN just/RB get/VB '
                                                        'out/IN of/IN hell/NN by/IN set/VBG up/RP an/DT end/NN of/IN '
                                                        'life/NN program/NN ?/.\nthis/DT dude/NN just/RB say/VBD '
                                                        'you/PRP would/MD go/VB to/IN hell/NN if/IN you/PRP do/VBD '
                                                        'not/RB buy/VB his/PRP$ service/NN ./.\nwhat/WDT a/DT '
                                                        'scumbag/NN ./.\n'},
              {'id': 'd25vklz', 'cat': 'Right', 'body': 'I/PRP have/VB see/VBN bad/JJR ./.\nlike/IN search/VBG "/\'\' '
                                                        'hillary/NNP clinton/NNP nude/NNP "/\'\' in/IN Google/NNP '
                                                        'image/NN search/NN ,/, even/RB with/IN safesearch/NN on/IN '
                                                        './.\n'}]
proc_left = [{'id': 'c5azjz5', 'cat': 'Left', 'body': 'worry/VBG about/IN who/WP say/VBD something/NN be/VBZ '
                                                      'appeal/VBG to/IN authority/NN ./.\nI/PRP like/VBP quote/NNS '
                                                      'that/WDT be/VBP relevant/JJ and/CC valuable/JJ on/IN '
                                                      'their/PRP$ face/NN ,/, no/RB matter/RB who/WP say/VBD them/PRP '
                                                      './. "/``\nthose/DT who/WP would/MD sacrifice/VB liberty/NN '
                                                      'for/IN security/NN deserve/VBP neither/CC "/`` -/: voltaire/NN '
                                                      '?/.\nBen/NNP Franklin/NNP ?/.\nI/PRP do/VBP not/RB care/VB '
                                                      './.\nI/PRP think/VBP it/PRP be/VBZ a/DT valid/JJ statement/NN '
                                                      './.\n'},
             {'id': 'c5azl7r', 'cat': 'Left', 'body': 'My/PRP$ point/NN be/VBZ that/IN the/DT quote/NN be/VBZ good/JJ '
                                                      'even/RB divorce/VBN from/IN Golda/NNP Meir/NNP ,/, '
                                                      'and/CC that/IN this/DT be/VBZ ostensibly/RB a/DT discussion/NN '
                                                      'about/IN the/DT quote/UH itself/PRP ./.\na/DT thread/NN '
                                                      'devote/VBN to/IN Meir/NNP (/-LRB- and/CC maybe/RB ,/, say/UH ,'
                                                      '/, Thatcher/NNP too/RB )/-RRB- would/MD be/VB a/DT good/JJ '
                                                      'thread/NN for/IN TwoX/NNP as/RB well/RB ,/, but/CC */NFP '
                                                      'it/PRP should/MD be/VB its/PRP$ own/JJ thread/NN ./. */NFP\n'},
             {'id': 'c5azniu', 'cat': 'Left', 'body': 'It/PRP sound/VBZ like/IN you/PRP be/VBP good/JJ at/IN '
                                                      'listen/VBG to/IN your/PRP$ body/NN when/WRB it/PRP come/VBZ '
                                                      'to/IN food-/IN too/RB many/JJ people/NNS (/-LRB- I/PRP be/VBP '
                                                      'guilty/JJ too/RB !/. )/-RRB-\nwill/MD just/RB keep/VB eat/VBG '
                                                      'when/WRB our/PRP$ stomach/NN be/VBZ full/JJ !/.\n'}]





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

    center_dir = os.path.join(args.a1_dir, 'feats/Center_IDs.txt')
    right_dir = os.path.join(args.a1_dir, 'feats/Right_IDs.txt')
    left_dir = os.path.join(args.a1_dir, 'feats/Left_IDs.txt')
    alt_dir = os.path.join(args.a1_dir, 'feats/Alt_IDs.txt')

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
    if comment_class == "Right":
        index = right_id.index(comment_id)
        liwc = right_feats[index]
        for i in range(29, 173):
            feat[i] = liwc[i-29]
    if comment_class == "Left":
        index = left_id.index(comment_id)
        liwc = left_feats[index]
        for i in range(29, 173):
            feat[i] = liwc[i-29]
    if comment_class == "Alt":
        index = alt_id.index(comment_id)
        liwc = alt_feats[index]
        for i in range(29, 173):
            feat[i] = liwc[i-29]
    print(feat)
    return feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                        default='/u/cs401/A1/')
    args = parser.parse_args()

    LIWC(args)
    IDs(args)
    feats = np.zeros(173)

    start = time.time()


    for comment in proc_center:
        extract2(feats, comment["cat"], comment["id"])
    for comment in proc_right:
        extract2(feats, comment["cat"], comment["id"])
    for comment in proc_left:
        extract2(feats, comment["cat"], comment["id"])

    end = time.time()
    print(f"Runtime of the program is {end - start}")



