import torch.nn as nn
import torch

class SimpleFactorRanker(nn.Module):
    def __init__(self, num_items, num_users, num_latent_factors):
        super(SimpleFactorRanker, self).__init__()
        self.num_items = num_items
        self.num_users = num_users
        self.num_latent_factors = num_latent_factors
        # we normalize the weights to have l2 norm 1 for each vector
        self.item_weights = nn.Embedding(self.num_items, self.num_latent_factors,
                                         max_norm=1.0)
        self.user_weights = nn.Embedding(self.num_users, self.num_latent_factors,
                                         max_norm=1.0)

    def forward(self,inputUsers,inputItems,negativeItems):
        iI = self.item_weights(inputItems)
        iU = self.user_weights(inputUsers)
        iN = self.item_weights(negativeItems)

        itemScore = torch.einsum('bl,bl->b',(iU,iI))
        negScore = torch.einsum('bl,bnl->bn',(iU,iN))

        return itemScore, negScore
