import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Fasttext_model(nn.Module):

    def __init__(self, embedding_dim,ft_model):
        super(Fasttext_model, self).__init__()
        self.embedding_dim = embedding_dim

        #ft_model = fasttext.util.reduce_model(ft, embedding_dim)

        self.ft_model = ft_model

        self.fc1 = nn.Linear(embedding_dim, 2*9*9*9)

    def forward(self,descriptions):
        embeddings = torch.zeros(len(descriptions),300).cuda()
        for i in range(len(descriptions)):
            embeddings[i] = torch.from_numpy(self.ft_model.get_sentence_vector(descriptions[i]))

        x = self.fc1(embeddings)

        return x



