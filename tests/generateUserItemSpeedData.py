import numpy as np
import sys
import time
import argparse
import h5py

parser = argparse.ArgumentParser(description='Generate User Item synthetic data')
parser.add_argument('-u','--users', dest = 'users',type=int, nargs=1,required=True)
parser.add_argument('-i','--items', dest = 'items',type=int,nargs=1,required=True)
parser.add_argument('-e','--examples', dest = 'examples', type = int, nargs=1,required=True)
parser.add_argument('-f','--dest_file', dest = 'dest_file', type = str,nargs=1,required=True)
args = parser.parse_args()

if __name__ == "__main__":
    t0 = time.time()

    np.random.seed(12)
    # sample users
    users = np.random.random_integers(low = 0, high = args.users[0],
                                      size = args.examples[0])\
                     .astype(np.int64)
    
    # sample items
    items = np.sort(np.reshape(np.random.random_integers(low = 0, high = args.items[0],
                                          size = args.examples[0]*3),[-1,3])\
                    .astype(np.int64),axis=1)

    dest_file =  h5py.File(args.dest_file[0], 'w')

    dset = {}
    dset['user'] = dest_file.create_dataset('user', data = users)
    dset['item'] = dest_file.create_dataset('item', data = items[:,1])
    dset['minItem'] = dest_file.create_dataset('minItem', data = items[:,0])
    dset['maxItem'] = dest_file.create_dataset('maxItem', data = items[:,0])

    # dset = dict([(name, (args.examples[0],), dtype=np.int64))
    #              for name in ('user','item','minItem','maxItem')])

    # idx = 0
    # for u, i1, i2, i3 in np.nditer([users,items[:,0],items[:,1],items[:,2]]):
    #     ilist = sorted([i1,i2,i3])
    #     dset['user'][idx] = u
    #     dset['item'][idx] = ilist[1]
    #     dset['minItem'][idx] = ilist[0]
    #     dset['maxItem'][idx] = ilist[2]
    #     idx += 1

    [dset[k].flush() for k in dset.keys()]
    dest_file.close()

    

    

    
