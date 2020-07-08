import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM_model(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, wv):
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_to_vector = wv

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim + 6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, sentence, cube_dimension):
        embeds = self.calc_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        x = self.fc1(torch.cat((lstm_out[-1].view(-1),cube_dimension)))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def calc_embeddings(self,sentence):
        words = sentence.split()
        seq_length = len(words)
        embeds = torch.zeros(seq_length,1,self.embedding_dim)
        for i,word in enumerate(words):
            embeds[i] = torch.from_numpy(self.word_to_vector[word]).view(1,1,-1)
        return embeds


