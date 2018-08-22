import numpy as np
import sys
import time
import argparse
import h5py

class ConvertToInt(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
         if nargs is not None:
             raise ValueError("nargs not allowed")
         super(ConvertToInt, self).__init__(option_strings, dest, **kwargs)
         
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(float(values)))
 
parser = argparse.ArgumentParser(description='Generate User Item synthetic data')
# parser.add_argument('-u','--users', dest = 'users',type=int, nargs=1,required=True)
# parser.add_argument('-i','--items', dest = 'items',type=int,nargs=1,required=True)
# parser.add_argument('-e','--examples', dest = 'examples', type = int, nargs=1,required=True)
# parser.add_argument('-f','--dest_file', dest = 'dest_file', type = str,nargs=1,required=True)
parser.add_argument('-u','--users',action=ConvertToInt)
parser.add_argument('-i','--items',action=ConvertToInt)
parser.add_argument('-e','--examples', action = ConvertToInt)
parser.add_argument('-f','--dest_file',dest='dest_file', type = str)

args = parser.parse_args()

if __name__ == "__main__":
    t0 = time.time()

    np.random.seed(12)
    # sample users
    users = np.random.random_integers(low = 0, high = args.users,
                                      size = args.examples)\
                     .astype(np.int64)
    
    # sample items
    items = np.sort(np.reshape(np.random.random_integers(low = 0, high = args.items,
                                          size = args.examples*3),[-1,3])\
                    .astype(np.int64),axis=1)

    dest_file =  h5py.File(args.dest_file, 'w')

    dset = {}
    dset['user'] = dest_file.create_dataset('user', data = users)
    dset['item'] = dest_file.create_dataset('item', data = items[:,1])
    dset['minItem'] = dest_file.create_dataset('minItem', data = items[:,0])
    dset['maxItem'] = dest_file.create_dataset('maxItem', data = items[:,0])

    [dset[k].flush() for k in dset.keys()]
    dest_file.close()

    

    

    
