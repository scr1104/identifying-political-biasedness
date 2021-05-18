#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz


import sys
import argparse
import os
import json
import re
import spacy
import html

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment, steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    if comment in ["[deleted]", "[removed]"]:
        return ""

    modComm = comment

    if 1 in steps:  # replace non-space whitespace characters
        modComm = re.sub(r"[\t\n\r]+", " ", modComm)

    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)

    if 4 in steps:  # remove duplicate spaces.
        modComm = re.sub(r"\s{2,}", " ", modComm)

    if 5 in steps:

        utt = nlp(modComm)

        mod = ""
        for sent in utt.sents:
            one_sentence = []

            for token in sent:
                if token.lemma_[0] == "-":
                    new_token = token.text + "/" + token.tag_
                else:
                    new_token = token.lemma_ + "/" + token.tag_

                one_sentence.append(new_token)

            mod += " ".join(one_sentence) + "\n"

        modComm = mod

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            cat = os.path.basename(fullFile)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            sampling = args.ID[0] % len(data)

            # select appropriate args.max lines
            lines = data[sampling: sampling + args.max]

            # read lines
            for line in lines:
                row = json.loads(line)

                # choose to retain fields from the lines that are relevant
                # keys = ["id", "controversiality", "score", "author", "subreddit"]
                # dic = dict((key, row[key]) for key in keys)
                # add a field to each selected line called 'cat' with the value of 'file'
                dic = {"id": row["id"], "cat": cat}

                # process the body field (row['body']) with preproc1() using default for `steps` argument
                comment = row["body"]

                # replace the 'body' field with the processed text
                dic["body"] = preproc1(comment)

                # append the result to 'allOutput'
                allOutput.append(dic)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir",
                        help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.",
                        default='/u/cs401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
