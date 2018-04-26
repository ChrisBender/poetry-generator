from arguments import get_arguments
from common import get_model_and_data

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import random
import time
import os


def train(net, criterion, optimizer, input, output, update_model=True):

    total_length = input.size()[0]
    # h = Variable(torch.zeros(num_layers, 1, hidden_size).type(float_tensor))
    h_word = (torch.zeros(net.word_num_layers, 1, net.word_hidden_size).type(net.float_tensor),
         torch.zeros(net.word_num_layers, 1, net.word_hidden_size).type(net.float_tensor))
    h_char = (torch.zeros(net.char_num_layers, 1, net.char_hidden_size).type(net.float_tensor),
         torch.zeros(net.char_num_layers, 1, net.char_hidden_size).type(net.float_tensor))

    h_word = (Variable(h_word[0]), Variable(h_word[1]))
    h_char = (Variable(h_char[0]), Variable(h_char[1]))

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
    net, poems_split_by_words = get_model_and_data(args)

    random.shuffle(poems_split_by_words)
    training_num_poems = int(len(poems_split_by_words) * args.training_testing_split)
    testing_num_poems = len(poems_split_by_words) - training_num_poems
    training_poems = poems_split_by_words[:training_num_poems]
    testing_poems = poems_split_by_words[training_num_poems:]
    
    training_poems = training_poems[:5]
    testing_poems = testing_poems[:1]
    testing_poems = training_poems
    
    print("*** Num training poems: {0} ***".format(training_num_poems))
    print("*** Num testing poems: {0} ***".format(testing_num_poems))
    print("*** Vocabulary size: {0} ***".format(len(net.word2i)))

    num_params = 0
    for param in list(net.parameters()):
        product_of_dimensions = 1
        for dimension in param.size():
            product_of_dimensions *= dimension
        num_params += product_of_dimensions
    print("*** Num Params: {0} ***".format(num_params))


    if not args.no_cuda and torch.cuda.is_available():
        print("Created net. Sending to GPU ...")
        net.cuda()
    print("GatedLSTM:", net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
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
            total_loss += train(net, criterion, optimizer, input, output)

        total_time += (time.time() - start_time)
    
        if epoch % args.print_example_every == args.print_example_every - 1:
            testing_losses = [
                    train(net, criterion, optimizer, 
                        split_poem[0], split_poem[1], update_model=False) 
                    for split_poem in split_testing_poems
            ]
            testing_loss = sum(testing_losses) / len(testing_losses)
    
            if args.verbose:
                print("*" * 30)

            print("(epoch, training_loss, testing_loss, time) = ({0}, {1}, {2}, {3})".format(
                epoch + 1,
                total_loss / (len(split_training_poems) * args.print_example_every),
                testing_loss, 
                total_time / args.print_example_every
            ))

            if args.verbose:
                print("Sample:")
                print(repr(net.get_sample()))
    
    checkpoint = {
            'model' : net.state_dict(), 
            'optimizer' : optimizer.state_dict(),
            'args' : args
    }

    if not os.path.exists("models/" + args.dataset):
            os.makedirs("models/" + args.dataset)
    torch.save(checkpoint, "models/" + args.dataset + "/model.pt")
    print("*** Saved model. ***")

if __name__ == "__main__":
    main()

