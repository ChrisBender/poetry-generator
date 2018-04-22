import argparse

import csv
import pickle
import nltk.tokenize.moses as moses

from arguments import get_arguments

def parse_corpus(dataset_name):

    dataset_path = "data/" + dataset_name + "/"
    try:
        dataset_file = open(dataset_path + "raw.csv")
    except OSError:
        print("Could not find the file " + dataset_path + "raw.csv. Terminating.")
        return

    print("*** Parsing raw corpus ... ***")

    tokenizer = moses.MosesTokenizer()
    poems_split_by_words = []
    word2i, i2word = {}, {}

    for poem in csv.reader(dataset_file):
        
        poem = poem[0]
        poem_split_by_words = ["SOP"]

        for line in poem.split('\n')[:-1]:
            tokens = []
            for token in tokenizer.tokenize(line, escape=False):
                
                assert len(token) > 0, "Empty token."
                if token[0].isupper():
                    tokens.append("CAP")
                
                # TODO: Support data like "\'When ... ". Currently, we lowercase every word.
                token = token.lower()
                tokens.append(token)

                if token not in word2i:
                    new_index = len(i2word)
                    word2i[token] = new_index
                    i2word[new_index] = token
            
            tokens.append('\n')
            poem_split_by_words.extend(tokens)

        poem_split_by_words.append("EOP")
        poems_split_by_words.append(poem_split_by_words)

    with open(dataset_path + "processed.pkl", 'wb') as file:
        pickle.dump(poems_split_by_words, file)
    with open(dataset_path + "index.pkl", 'wb') as file:
        pickle.dump(word2i, file)
        pickle.dump(i2word, file)

    print("*** Finished parsing corpus. ***")


def main():
    args = get_arguments('parser') 
    parse_corpus(args.dataset_name)


if __name__ == "__main__":
    main()    

