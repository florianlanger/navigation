import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_model(nn.Module):

    def __init__(self, embedding_dim, hidden_dim,len_vocab):
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(len_vocab,embedding_dim,padding_idx=0)
        #self.word_to_vector = wv

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)#num_layers = 2,dropout = 0.3)

        self.fc1 = nn.Linear(hidden_dim, 2*9*9*9)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 9*9*9)


    def forward(self,cube_dimensions,descriptions,length_descriptions):
        embeds = self.embedding(descriptions)
        packed_input = pack_padded_sequence(embeds, length_descriptions.numpy(), batch_first=True,enforce_sorted=False)

        packed_output, (ht, ct) = self.lstm(packed_input)

        # only interested in output after last time step
        # squeeze extra first dim
        ht = torch.squeeze(ht)
        #x = self.fc1(torch.cat((lstm_out[-1].view(-1),cube_dimensions)))
        x = self.fc1(ht)

        #x = torch.sigmoid(x)
        return x

    # def calc_embeddings(self,sentences):
    #     max_number_words = max([len(sentence.split()) for sentence in sentences])
    #     embeds = torch.zeros(len(sentences),max_number_words,self.embedding_dim)
    #     for i,sentence in enumerate(sentences):
    #         for j, word in enumerate(sentence.split()):
    #             embeds[i,j] = self.embedding(word)
    #     return embeds


