import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import nltk.tokenize.moses as moses


class GatedLSTM(nn.Module):

    def __init__(self, word2i, i2word, glove_embedding, chars, word_hidden_size, char_hidden_size,
                 word_num_layers, char_num_layers, dropout, use_cuda):

        super().__init__()

        self.tags = ('CAP', 'EOP')
        self.detokenizer = moses.MosesDetokenizer()

        assert type(word2i) == dict and type(i2word) == dict and len(word2i) == len(i2word), \
            'Malformed word lookup tables.'

        if use_cuda:
            self.float_tensor = torch.cuda.FloatTensor
            self.long_tensor = torch.cuda.LongTensor
        else:
            self.float_tensor = torch.FloatTensor
            self.long_tensor = torch.LongTensor

        self.word2i = word2i
        self.i2word = i2word
        self.glove_embedding = glove_embedding.type(self.float_tensor)
        self.chars = chars
        self.word_hidden_size = word_hidden_size
        self.char_hidden_size = char_hidden_size
        self.word_num_layers = word_num_layers
        self.char_num_layers = char_num_layers
        self.word_vocab_size = len(word2i)
        self.char_vocab_size = len(chars) + len(self.tags)

        # self.word_encoder = nn.Embedding(self.word_vocab_size, self.word_hidden_size)
        self.word_encoder = lambda x_word: self.glove_embedding[x_word, :]
        self.word_lstm = nn.LSTM(self.word_hidden_size, self.word_hidden_size, self.word_num_layers, dropout = dropout)
        self.word_decoder = nn.Linear(self.word_hidden_size, self.word_vocab_size)

        self.char_encoder = nn.Embedding(self.char_vocab_size, self.char_hidden_size)
        self.char_lstm = nn.LSTM(self.char_hidden_size, self.char_hidden_size, self.char_num_layers, dropout = dropout)
        # self.char_decoder = nn.Linear(self.char_hidden_size, self.char_vocab_size)

        self.char_to_embedding = nn.Parameter(torch.randn(self.word_hidden_size, self.char_hidden_size)).type(self.float_tensor)
        self.x_word_to_g_weight = nn.Parameter(torch.randn(self.word_hidden_size)).type(self.float_tensor)
        self.x_word_to_g_bias = nn.Parameter(torch.randn(1)).type(self.float_tensor)

    def forward(self, x_word, h_word, h_char):

        word = self.i2word[x_word.data[0]]
        x_char = self.chars_to_tensor(word)

        for c in x_char:
            c = self.char_encoder(c.view(1, -1))
            _, h_char = self.char_lstm(c, h_char)
        # x_char = self.char_decoder(h_char)
        x_char = torch.matmul(self.char_to_embedding, h_char[1][-1,:,:].t())

        x_word = self.word_encoder(x_word.view(1, -1))

        g = F.relu(torch.dot(self.x_word_to_g_weight, x_word[0,0,:]) + self.x_word_to_g_bias)
        x_word = x_word.squeeze(0)
        x_char = x_char.t()

        x = (1 - g) * x_word + g * x_char
        x = x.view(1, 1, self.word_hidden_size)

        y, h_word = self.word_lstm(x, h_word)
        y = self.word_decoder(y.squeeze(0))
        return y, h_word, h_char

    def words_to_tensor(self, words):
        tensor = torch.zeros(len(words)).type(self.long_tensor)
        for i in range(len(words)):
            assert words[i] in self.word2i, "Failed to recognize word {0} in word2i.".format(words[i])
            tensor[i] = self.word2i[words[i]]
        return Variable(tensor)

    def chars_to_tensor(self, string):

        assert " " not in string, "Input string {0} is not a single word.".format(string)

        if string in self.tags:
            tensor = torch.zeros(1).type(self.long_tensor)
            tensor[0] = self.char_vocab_size - len(self.tags) + self.tags.index(string)
        else:
            tensor = torch.zeros(len(string)).type(self.long_tensor)
            for i in range(len(string)):
                assert string[i] in self.chars, 'character {0} not found in self.chars = {1}.'.format(string[i], self.chars)
                tensor[i] = self.chars.index(string[i])

        return Variable(tensor)

    def get_sample(self, init_string = "\n", max_length = 250, temperature = 0.8):

        init_words = init_string.split(' ')

        assert all([word in self.word2i for word in init_words]), \
            'The initial string ({0}) contains word(s) not in the vocabulary.'.format(init_string)

        h_word = (Variable(torch.zeros(self.word_num_layers, 1, self.word_hidden_size).type(self.float_tensor)),
             Variable(torch.zeros(self.word_num_layers, 1, self.word_hidden_size).type(self.float_tensor)))
        h_char = (Variable(torch.zeros(self.char_num_layers, 1, self.char_hidden_size).type(self.float_tensor)),
             Variable(torch.zeros(self.char_num_layers, 1, self.char_hidden_size).type(self.float_tensor)))

        init_tensor = self.words_to_tensor(init_words)
        raw_predicted_words = init_words

        # Prime the network.
        for p in range(len(init_words) - 1):
            _, h_word, h_char = self.forward(init_tensor[p], h_word, h_char)

        # Build the prediction.
        x = init_tensor[-1]
        p = 0
        while p < max_length - len(init_words) and self.i2word[x.data[0]] != "EOP":
            y, h_word, h_char = self.forward(x, h_word, h_char)
            y_dist = y.data.view(-1).div(temperature).exp()
            chosen_i = torch.multinomial(y_dist, 1)[0]
            predicted_word = self.i2word[chosen_i]
            raw_predicted_words.append(predicted_word)
            x = self.words_to_tensor(predicted_word.split(' '))
            p += 1

        if raw_predicted_words[-1] == "EOP":
            raw_predicted_words.pop()

        detagged_predicted_words = []
        previous_word = ""

        for word in raw_predicted_words:
            assert word != "EOP", "EOP within predicted words."
            if word == "CAP":
                pass
            elif previous_word == "CAP":
                detagged_predicted_words.append(word[0].capitalize() + word[1:])
            else:
                detagged_predicted_words.append(word)
            previous_word = word


        indices_of_linebreaks = [i for i, word in enumerate(detagged_predicted_words) if word == '\n']

        if len(indices_of_linebreaks) > 0:
            detokenized_predicted_words = detagged_predicted_words[:indices_of_linebreaks[0] + 1]
            for i in range(len(indices_of_linebreaks) - 1):
                first = indices_of_linebreaks[i] + 1
                second = indices_of_linebreaks[i + 1]
                detokenized_selection = self.detokenizer.detokenize(detagged_predicted_words[first : second])
                detokenized_predicted_words.extend(detokenized_selection)
                detokenized_predicted_words.append('\n')
            detokenized_predicted_words.extend(detagged_predicted_words[indices_of_linebreaks[-1] + 1:])
        else:
            detokenized_predicted_words = self.detokenizer.detokenize(detagged_predicted_words.copy())

        return " ".join(detokenized_predicted_words).strip()

