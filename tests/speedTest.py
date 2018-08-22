import time
import argparse
import numpy as np
import h5py
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from aschioppa_torch_layers.NegativeSamplers import ShardedNegUniformSampler as SNU
from aschioppa_torch_layers.ranking import SimpleFactorRanker as SFR

class ConvertToInt(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
         if nargs is not None:
             raise ValueError("nargs not allowed")
         super(ConvertToInt, self).__init__(option_strings, dest, **kwargs)
         
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(float(values)))

parser = argparse.ArgumentParser(description='Generate User Item synthetic data')
parser.add_argument('-u','--users',action=ConvertToInt)
parser.add_argument('-i','--items',action=ConvertToInt)
parser.add_argument('-k','--num_latent', dest = 'num_latent', type = int, nargs=1, required=True)
parser.add_argument('-f','--source_file', dest = 'source_file', type = str,nargs=1,required=True)
parser.add_argument('-b','--batch_size', dest = 'batch_size', type = int,nargs=1,required=True)
parser.add_argument('-n','--num_negs', dest = 'num_negs', type = int,nargs=1,required=True)

args = parser.parse_args()



class H5Dataset(data.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data_user = h5_file.get('user')[:]
        self.data_item = h5_file.get('item')[:]
        self.data_minItem = h5_file.get('minItem')[:]
        self.data_maxItem = h5_file.get('maxItem')[:]

    def __getitem__(self, index):
        return (self.data_user[index],
                self.data_item[index],
                self.data_minItem[index],
                self.data_maxItem[index])

    def __len__(self):
        return self.data_user.shape[0]


if __name__ == "__main__":

    num_users = args.users
    num_items = args.items
    num_latent = args.num_latent[0]

    myDataset = H5Dataset(args.source_file[0])
    myDataLoader = DataLoader(myDataset,batch_size=args.batch_size[0],num_workers=1)
    #myDataLoader = DataLoader(myDataset,batch_size=1,num_workers=10)
    mySampler = SNU(args.num_negs[0])
    myProd = SFR(num_items+1,num_users+1,num_latent)
    lossFun = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(myProd.parameters(), lr=1e-2)
    #optimizer = torch.optim.Adam(myProd.parameters(), lr=1e-2,weight_decay=.001)
    print('Will start training')
    t0 = time.time()
    #idx = 0
    lossFun = torch.nn.MSELoss()
    for example in myDataLoader:
        # if idx % 100 == 0:
        #     print(time.time()-t0, '===>', idx)
        
        user, item, minItem, maxItem = example
        negSample = mySampler.getSample(item,minItem,maxItem)
        itemScore, negScore = myProd.forward(user,item,negSample)
        lossBpr = myProd.bpr(itemScore,negScore)
        #lossBpr = lossFun(itemScore,negScore[:,0])
        optimizer.zero_grad()
        lossBpr.backward()
        optimizer.step()
        # idx += 1
        # if idx > 1000:
        #     break
    print('End benchmark', time.time()-t0)

            
        
    
