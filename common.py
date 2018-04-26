from model import GatedLSTM
import pickle
import torch

chars_vocabulary = "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&\'()*+,-./:;—<=>?@[\\]^_`{|}~ " \
        + "\t\n\r\x0b\x0c"

def get_model_and_data(train_args):

    dataset_path = "data/" + train_args.dataset + "/"

    try:
        with open(dataset_path + "processed.pkl", 'rb') as file:
            poems_split_by_words = pickle.load(file)
        with open(dataset_path + "index.pkl", 'rb') as file:
            word2i = pickle.load(file)
            i2word = pickle.load(file)
    except OSError:
        print("Failed to load processed data. Ensure that you have run corpus_parser.py")

    assert len(word2i) == len(i2word)

    if train_args.use_glove:
        with open(dataset_path + "glove.pkl", 'rb') as file:
            glove_dict = pickle.load(file)
        l_embed, r_embed = glove_dict['l_embed'], glove_dict['r_embed']
        l_embed, r_embed = torch.stack(l_embed).squeeze(2), torch.stack(r_embed).squeeze(2)
        glove_embeddings = l_embed + r_embed
    else:
        glove_embeddings = None

    use_cuda = not train_args.no_cuda and torch.cuda.is_available()
    net = GatedLSTM(train_args, word2i, i2word, glove_embeddings, chars_vocabulary, use_cuda) 

    if use_cuda:
        net.cuda()

    return net, poems_split_by_words

