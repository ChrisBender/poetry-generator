from torch.autograd import Variable
import numpy as np
import torch
import torch.optim as optim
import pickle

# Set parameters
epochs = 30
learning_rate = 0.001
batch_size = 20
context_size = 10
embed_size = 300
x_max = 100
alpha = 0.75

def train_glove():

    # Open and read in text
    with open('./data/data_processed.txt', 'rb') as file:
        poems_split_by_words = pickle.load(file)


    # Create word list and vocabulary
    # poems = [" ".join(poem) for poem in poems_split_by_words]
    # words = " ".join(poems)
    words = []
    for poem in poems_split_by_words:
        words += poem

    word2i, i2word = {}, {}
    for poem in poems_split_by_words:
        for word in poem:
            if word not in word2i:
                new_index = len(i2word)
                word2i[word] = new_index
                i2word[new_index] = word

    vocab = word2i.keys()
    num_words = len(words)
    vocab_size = len(vocab)


    # Construct co-occurrence matrix
    cooccurrences = np.zeros((vocab_size, vocab_size))
    for i in range(num_words):
        for j in range(1, context_size + 1):
            index = word2i[words[i]]
            if i - j >= 0:
                left_index = word2i[words[i-j]]
                cooccurrences[index, left_index] += 1.0 / j
            if i + j < num_words:
                right_index = word2i[words[i+j]]
                cooccurrences[index, right_index] += 1.0 / j

    # Get indices of non-zero co-occurrences
    cooccurrence_indices = np.transpose(np.nonzero(cooccurrences))

    # Weight function
    def sigma(x):
        if x < x_max:
            return (x / x_max) ** alpha
        return 1

    # Set up word vectors and biases
    l_embed, r_embed = [
        [Variable(torch.from_numpy(np.random.normal(0, 0.01, (embed_size, 1))),
            requires_grad=True) for _ in range(vocab_size)] for _ in range(2)]
    l_biases, r_biases = [
        [Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)),
            requires_grad=True) for _ in range(vocab_size)] for _ in range(2)]
    # l_embed, r_embed = [Variable(torch.from_numpy(np.random.normal(0, 0.01, (embed_size, vocab_size))), requires_grad=True)
    #                     for _ in range(2)]
    # l_biases, r_biases = [Variable(torch.from_numpy(np.random.normal(0, 0.01, (1, vocab_size))), requires_grad=True)
    #                       for _ in range(2)]


    # Set up optimizer
    optimizer = optim.Adam(l_embed + r_embed + l_biases + r_biases, lr=learning_rate)


    # Batch sampling function
    def gen_batch():

        sample = np.random.choice(np.arange(len(cooccurrence_indices)), size=batch_size, replace=False)
        l_vecs, r_vecs, covals, l_v_bias, r_v_bias = [], [], [], [], []

        for chosen in sample:
            index = tuple(cooccurrence_indices[chosen])
            l_vecs.append(l_embed[index[0]])
            r_vecs.append(r_embed[index[1]])
            covals.append(cooccurrences[index])
            l_v_bias.append(l_biases[index[0]])
            r_v_bias.append(r_biases[index[1]])

        return l_vecs, r_vecs, covals, l_v_bias, r_v_bias


    # Train model
    for epoch in range(epochs):

        num_batches = int(num_words/batch_size)
        avg_loss = 0.0

        for batch in range(num_batches):
            optimizer.zero_grad()
            l_vecs, r_vecs, covals, l_v_bias, r_v_bias = gen_batch()
            loss = sum(
                [torch.mul((torch.dot(l_vecs[i], r_vecs[i]) + l_v_bias[i] + r_v_bias[i] - np.log(covals[i])) ** 2,
                           sigma(covals[i]))
                 for i in range(batch_size)]
            )
            avg_loss += loss.data[0] / num_batches
            loss.backward()
            optimizer.step()

        print("Average loss for epoch {0} of {1} is {2}.".format(epoch + 1, epochs, avg_loss))


    # Save model
    print("*** Saving GloVe embedding ... ***")
    with open('./models/glove', 'wb') as file:
        glove_dict = {
            'l_embed' : l_embed,
            'r_embed' : r_embed,
            'l_biases' : l_biases,
            'r_biases' : r_biases,
            'word2i' : word2i,
            'i2word' : i2word
        }
        pickle.dump(glove_dict, file)
    print("*** Saved GloVe embedding. ***")

