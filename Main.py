from GatedLSTM import GatedLSTM
from CorpusParser import parse_corpus
from GloveEmbedding import train_glove

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import random
import time
import pickle

epochs = 25
learning_rate = 0.001
dropout = 0.9
# word_hidden_size = 512
# char_hidden_size = 256
# word_num_layers = 3
# char_num_layers = 3
word_hidden_size = 126
char_hidden_size = 126
word_num_layers = 2
char_num_layers = 1
print_example_every = 1
training_testing_split = 0.90
use_pretrained_model = False
use_cuda_when_available = False
reparse_corpus = True
retrain_glove = True
write_losses_to_file = True
quiet = False

print("* dout = {0} *".format(dropout))
print("* word_hidden = {0} *".format(word_hidden_size))

if reparse_corpus:
    parse_corpus()

if retrain_glove:
    train_glove()


use_cuda = use_cuda_when_available and torch.cuda.is_available() and not use_pretrained_model
if use_cuda:
    print("*** Using GPU acceleration. ***")
else:
    print("*** Not using GPU acceleration. ***")

with open('./data/data_processed.txt', 'rb') as file:
    poems_split_by_words = pickle.load(file)
word2i, i2word = {}, {}

for poem in poems_split_by_words:
    for word in poem:
        if word not in word2i:
            new_index = len(i2word)
            word2i[word] = new_index
            i2word[new_index] = word

random.shuffle(poems_split_by_words)
training_num_poems = int(len(poems_split_by_words) * training_testing_split)
training_poems = poems_split_by_words[:training_num_poems]
testing_poems = poems_split_by_words[training_num_poems:]

# training_poems = training_poems[:5]
# testing_poems = testing_poems[:1]
# testing_poems = training_poems

assert len(word2i) == len(i2word)
num_distinct_words = len(word2i)

training_num_poems = len(training_poems)
testing_num_poems = len(testing_poems)

print("*** Num training poems: {0} ***".format(training_num_poems))
print("*** Num testing poems: {0} ***".format(testing_num_poems))
print("*** Vocabulary size: {0} ***".format(num_distinct_words))


def split_poem(poem):
    input = net.words_to_tensor(poem[:-1])
    output = net.words_to_tensor(poem[1:])
    return input, output

def train(input, output, update_model=True):

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


chars_vocabulary = "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c"
net = GatedLSTM(word2i, i2word, chars_vocabulary, word_hidden_size, char_hidden_size,
                word_num_layers, char_num_layers, dropout, use_cuda)

num_params = 0
for param in list(net.parameters()):
    product_of_dimensions = 1
    for dimension in param.size():
        product_of_dimensions *= dimension
    num_params += product_of_dimensions
print("*** Num Params: {0} ***".format(num_params))

if use_pretrained_model:
    print("*** Loading pretrained model ... ***")
    checkpoint = torch.load('./models/lstm.pt')
    net.load_state_dict(checkpoint['model'])
    print("*** Loaded pretrained model. ***")
    for temp in (0.8,):
        for _ in range(3):
            print("*" * 30)
            print("*** Sample for temp = {0}: ***".format(temp))
            print(net.get_sample(max_length=1000, temperature=temp))
    exit(0)

if use_cuda:
    print("Created net. Sending to GPU ...")
    net.cuda()
print("RecurrentNetWord:", net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)

print("*** Splitting poems ... ***")

split_training_poems = [split_poem(poem) for poem in training_poems]
split_testing_poems = [split_poem(poem) for poem in testing_poems]

print("*** Split poems. ***")

total_loss, total_time = 0, 0

for epoch in range(epochs):

    start_time = time.time()

    i = 0
    for input, output in split_training_poems:
        i += 1
        if not quiet and (training_num_poems < 10 or i % (training_num_poems // 10) == 0):
            print("{0}% done for training epoch {1} of {2}.".format(int(100 * i / training_num_poems), epoch + 1, epochs))
        total_loss += train(input, output)

    total_time += (time.time() - start_time)

    if epoch % print_example_every == print_example_every - 1:

        testing_losses = [train(split_poem[0], split_poem[1], update_model=False) for split_poem in split_testing_poems]
        testing_loss = sum(testing_losses) / len(testing_losses)

        if not quiet:
            print("*" * 90)
        print("(epoch, training_loss, testing_loss, time) = ({0}, {1}, {2}, {3})"
              .format(epoch + 1, total_loss / (len(split_training_poems) * print_example_every),
                      testing_loss, total_time / print_example_every))
        if not quiet:
            sample = net.get_sample()
            # print(repr(sample))
            print(sample)
        total_loss, total_time = 0, 0



print("*** Saving model ... ***")
checkpoint = {'model' : net.state_dict(), 'optimizer' : optimizer.state_dict()}
torch.save(checkpoint, './models/lstm.pt')
print("*** Saved model. ***")

print("\n*** Final Sample: ***\n")
for _ in range(10):
    print("*" * 90)
    print(net.get_sample(max_length=1000))
