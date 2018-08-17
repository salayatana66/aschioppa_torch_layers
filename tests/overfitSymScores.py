import time
import argparse
import numpy as np
import h5py
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from aschioppa_torch_layers.NegativeSamplers import ShardedNegUniformSampler as SNU
from aschioppa_torch_layers.ranking import SimpleFactorRanker as SFR

parser = argparse.ArgumentParser(description='Generate User Item synthetic data')
parser.add_argument('-u','--users', dest = 'users',type=int, nargs=1,required=True)
parser.add_argument('-i''--items', dest = 'items',type=int,nargs=1,required=True)
parser.add_argument('-k','--num_latent', dest = 'num_latent', type = int, nargs=1, required=True)
parser.add_argument('-f','--source_file', dest = 'source_file', type = str,nargs=1,required=True)
args = parser.parse_args()



class H5Dataset(data.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('user_item')
        self.target = h5_file.get('score')

    def __getitem__(self, index):
        return (self.data[index,:],self.target[index])

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":

    num_users = args.users[0]
    num_items = args.items[0]
    num_latent = args.num_latent[0]

    myNegSize = 1
    
    myDataset = H5Dataset(args.source_file[0])
    myDataLoader = DataLoader(myDataset,batch_size=5,shuffle=True)

    mySampler = SNU(myNegSize)
    myProd = SFR(num_items,num_users,num_latent)
    lossFun = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(myProd.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(myProd.parameters(), lr=1e-2)
    for iteration in range(1000):
        aggLoss = 0
        numExa = 0
        for example in myDataLoader:
            optimizer.zero_grad()
            user_items = example[0]
            score = example[1]

            negSample = mySampler.getSample(user_items[:,1],torch.zeros_like(user_items[:,1]),
                                           num_items*torch.ones_like(user_items[:,1])-1)
            itemScore, negScore = myProd.forward(user_items[:,0],user_items[:,1],negSample)
            loss = lossFun(itemScore, score)
            loss.backward()
            optimizer.step()
            # we don't care about the negative score
            aggLoss += loss
            numExa += 1
        print(iteration, aggLoss/numExa)
            
        
    
