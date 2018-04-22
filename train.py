from model import GatedLSTM

from arguments import get_arguments

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import random
import time
import pickle


chars_vocabulary = "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ " \
        + "\t\n\r\x0b\x0c"


def train(net, input, output, update_model=True):

    total_length = input.size()[0]
    # h = Variable(torch.zeros(num_layers, 1, hidden_size).type(float_tensor))
    h_word = (Variable(torch.zeros(word_num_layers, 1, word_hidden_size).type(net.float_tensor)),
         Variable(torch.zeros(word_num_layers, 1, word_hidden_size).type(net.float_tensor)))
    h_char = (Variable(torch.zeros(char_num_layers, 1, char_hidden_size).type(net.float_tensor)),
         Variable(torch.zeros(char_num_layers, 1, char_hidden_size).type(net.float_tensor)))

    if update_model:
        net.zero_grad()

    loss = 0
    for c in range(total_length):
        y, h_word, h_char = net.forward(input[c], h_word, h_char)
        loss += criterion(y, output[c])

    if update_model:
        loss.backward()
        optimizer.step()

    return loss.data[0] / total_length



def main():

    args = get_arguments('train')
    
    use_cuda = not args.no_cuda and torch.cuda.is_available() 
    if use_cuda:
        print("*** Using GPU acceleration. ***")
    else:
        print("*** Not using GPU acceleration. ***")

    dataset_path = "data/" + args.dataset_name + "/"
    model_path = "models/" + args.dataset_name + "/"

    with open(dataset_path + "processed.pkl", 'rb') as file:
        poems_split_by_words = pickle.load(file)
    with open(dataset_path + "index.pkl", 'rb') as file:
        word2i = pickle.load(file)
        i2word = pickle.load(file)

    assert len(word2i) == len(i2word)
    
    random.shuffle(poems_split_by_words)
    training_num_poems = int(len(poems_split_by_words) * training_testing_split)
    testing_num_poems = len(poems_split_by_words) - training_num_poems
    training_poems = poems_split_by_words[:training_num_poems]
    testing_poems = poems_split_by_words[training_num_poems:]
    
    #training_poems = training_poems[:5]
    #testing_poems = testing_poems[:1]
    #testing_poems = training_poems
    
    print("*** Num training poems: {0} ***".format(training_num_poems))
    print("*** Num testing poems: {0} ***".format(testing_num_poems))
    print("*** Vocabulary size: {0} ***".format(len(word2i)))

    if args.use_glove:
        with open('./models/glove', 'rb') as file:
            glove_dict = pickle.load(file)
        
        l_embed, r_embed = glove_dict['l_embed'], glove_dict['r_embed']
        l_embed, r_embed = torch.stack(l_embed).squeeze(2), torch.stack(r_embed).squeeze(2)
        glove_embeddings = l_embed + r_embed
    else:
        glove_embeddings = None

    
    net = GatedLSTM(args, word2i, i2word, glove_embeddings, chars_vocabulary, use_cuda) 

    num_params = 0
    for param in list(net.parameters()):
        product_of_dimensions = 1
        for dimension in param.size():
            product_of_dimensions *= dimension
        num_params += product_of_dimensions
    print("*** Num Params: {0} ***".format(num_params))


    if use_cuda:
        print("Created net. Sending to GPU ...")
        net.cuda()
    print("GatedLSTM:", net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    split_training_poems = [net.split_poem(poem) for poem in training_poems]
    split_testing_poems = [net.split_poem(poem) for poem in testing_poems]
    
    for epoch in range(args.epochs):
    
        total_loss, total_time = 0, 0
        start_time = time.time()
    
        i = 0
        for input, output in split_training_poems:
            i += 1
            if args.verbose and (training_num_poems < 10 or i % (training_num_poems // 10) == 0):
                print("{0}% done for training epoch {1} of {2}.".format(
                    int(100 * i / training_num_poems), epoch + 1, args.epochs)
                )
            total_loss += train(net, input, output)

        total_time += (time.time() - start_time)
    
        if epoch % print_example_every == print_example_every - 1:
            testing_losses = [
                    train(net, split_poem[0], split_poem[1], update_model=False) 
                    for split_poem in split_testing_poems
            ]
            testing_loss = sum(testing_losses) / len(testing_losses)
    
            if args.verbose:
                print("*" * 30)

            print("(epoch, training_loss, testing_loss, time) = ({0}, {1}, {2}, {3})".format(
                epoch + 1,
                total_loss / (len(split_training_poems) * print_example_every),
                testing_loss, 
                total_time / print_example_every
            ))

            if args.verbose:
                print("Sample:")
                print(net.get_sample())
    
    checkpoint = {'model' : net.state_dict(), 'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, model_path + "model.pt")
    print("*** Saved model. ***")

if __name__ == "__main__":
    main()

