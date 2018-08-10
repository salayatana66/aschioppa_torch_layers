import torch
from aschioppa_torch_layers.NegativeSamplers import ShardedNegUniformSampler as SNU
from aschioppa_torch_layers.ranking import SimpleFactorRanker as SFR

def generate_sample(low,high_item,high_user,size):
    # (size, 3) tensor
    # min Id, selected Id, max Id
    return {"user" : torch.randint(low=0,high=high_user,size=(size,)).type(torch.LongTensor),
        "item" : (torch.randint(low=low,
                          high=high_item,size=(size,3))
            .sort(dim=1))[0].type(torch.LongTensor)
            }

if __name__ == "__main__":
    mySeed = 17
    torch.manual_seed(mySeed)
    print("Chosen seed: ", mySeed)
    myIter = 15
    print("Number of iterations for the test: ", myIter)
    myLow = 0
    myHighItem = 100
    print("Item range: [",myLow,", ",myHighItem,")")
    myHighUser = 50
    print("User range: [",myLow,", ",myHighUser,")")
    myBatchSize = 5
    print("Number of samples per batch: ", myBatchSize)
    myEmbSize = 7
    print("Embedding size: ", myEmbSize)
    myNegSize = 3
    print("Number of negatives per positive item: ", myNegSize)
    mySampler = SNU(myNegSize)
    myProd = SFR(myHighItem,myHighUser,myEmbSize)

    mySample = generate_sample(myLow,myHighItem,myHighUser,myBatchSize)
    for ii in range(myIter):
        print("Iter: ", ii)
        minItem = mySample['item'][:,0]
        Item = mySample['item'][:,1]
        maxItem = mySample['item'][:,2]

        negSample = mySampler.getSample(Item,minItem,maxItem)
        itemScore, negScore = myProd.forward(mySample['user'],Item,negSample)
        print(itemScore)
        print(negScore)
