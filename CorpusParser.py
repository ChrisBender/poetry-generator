import csv
import pickle
import nltk.tokenize.moses as moses

def parse_corpus():

    print("*** Parsing raw corpus ... ***")

    tokenizer = moses.MosesTokenizer()
    poems_split_by_words = []

    for poem in csv.reader(open('./data/data_raw.csv')):
        poem = poem[0]
        poem_split_by_words = []
        for line in poem.split('\n')[:-1]:
            tokens = []
            for token in tokenizer.tokenize(line, escape=False):
                assert len(token) > 0, "Empty token."
                if token[0].isupper():
                    tokens.append("CAP")
                    tokens.append(token.lower())
                else:
                    # TODO: Support data like "\'When ... ". Currently, we lowercase every word.
                    tokens.append(token.lower())
            tokens.append('\n')
            poem_split_by_words.extend(tokens)
        poem_split_by_words.append("EOP")
        poems_split_by_words.append(poem_split_by_words)

    with open("./data/data_processed.txt", 'wb') as file:
        pickle.dump(poems_split_by_words, file)

    print("*** Finished parsing corpus. ***")

