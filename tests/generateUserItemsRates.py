import time
import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser(description='Generate User Item synthetic data')
parser.add_argument('-u','--users', dest = 'users',type=int, nargs=1,required=True)
parser.add_argument('-i''--items', dest = 'items',type=int,nargs=1,required=True)
parser.add_argument('-k','--num_latent', dest = 'num_latent', type = int, nargs=1, required=True)
parser.add_argument('-f','--dest_file', dest = 'dest_file', type = str,nargs=1,required=True)
args = parser.parse_args()

if __name__ == "__main__":
    # set standard seed
    np.random.seed(0)

    # generate user & items latent factors
    U = np.random.random((args.users[0],args.num_latent[0])).astype(np.float32)
    I = np.random.random((args.num_latent[0],args.items[0])).astype(np.float32)

    # compute scorings
    R = U.dot(I).astype(np.float32)
    num_users = R.shape[0]
    num_items = R.shape[1]
    # open h5 file
    dest_file =  h5py.File(args.dest_file[0], 'w')
    dset1 = dest_file.create_dataset("user_item", (num_users*num_items,2), dtype=np.int64)
    dset2 = dest_file.create_dataset("score", (num_users*num_items,), dtype="f")
    # for loop over R
    it = np.nditer(R, flags=['multi_index'])
    while not it.finished:
        # extract user,item,score
        uu = it.multi_index[0]
        ii = it.multi_index[1]
        ss = it[0]
        dset1[uu*num_items+ii,0] = uu
        dset1[uu*num_items+ii,1] = ii
        dset2[uu*num_items+ii] = ss
        it.iternext()

    dset1.flush()
    dset2.flush()
    dest_file.close()    
