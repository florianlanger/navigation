import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, ft_model,mode):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)
        self.ft_model = ft_model
        self.mode = mode

        self.lstm = nn.LSTM(dim_in, dim_in)


    def forward(self,descriptions,length_descriptions):
        if self.mode == "debug":
            embeddings = torch.rand((len(descriptions),40,300))
        elif self.mode == "normal":
            embeddings = torch.zeros(len(descriptions),40,300)
            for i in range(len(descriptions)):
                split_descriptions = descriptions[i].split()
                for j in range(len(split_descriptions)):
                    embeddings[i,j] = torch.from_numpy(self.ft_model.get_word_vector(split_descriptions[j]))

        packed_input = pack_padded_sequence(embeddings.cuda(), length_descriptions, batch_first=True,enforce_sorted=False)

        packed_output, (ht, ct) = self.lstm(packed_input)

        # only interested in output after last time step
        # squeeze extra first dim
        ht = torch.squeeze(ht)
        return self.pi_network(ht), self.normal_network(ht)

    def loss(self, x1,x2, y):
        pi, normal = self.forward(x1,x2)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)



