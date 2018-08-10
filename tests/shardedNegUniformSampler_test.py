import torch
from aschioppa_torch_layers.NegativeSamplers import ShardedNegUniformSampler as SNU

def generate_sample(low,high,size):
    # (size, 3) tensor
    # min Id, selected Id, max Id
    return (torch.randint(low=low,
                          high=high,size=(size,3))
            .sort(dim=1))[0]

if __name__ == "__main__":
    mySeed = 17
    torch.manual_seed(mySeed)
    print("Chosen seed: ", mySeed)
    myIter = 15
    print("Number of iterations for the test: ", myIter)
    myLow = 0
    myHigh = 5
    print("Iter range: [",myLow,", ",myHigh,")")
    mySize = 10
    print("Number of samples per iteration: ", mySize)
    mySampler = SNU(mySize)

    for ii in range(myIter):
        print("Iteration ", ii)
        x = generate_sample(myLow,myHigh,mySize).type(torch.LongTensor)
        minIds = x[:,0]
        inputIds = x[:,1]
        maxIds = x[:,2]

        currSample = mySampler.getSample(inputIds, minIds, maxIds)
        # check the sample is between min & max
        assert (currSample <= maxIds.view(-1,1)).sum() == maxIds.size()[0]*mySize
        assert (currSample >= minIds.view(-1,1)).sum() == minIds.size()[0]*mySize
        # still collisions because can happen that max & min are equal
        # this is the correct invariant
        assert ( (currSample != inputIds.view(-1,1)) |
                  (maxIds.view(-1,1).expand(-1,mySize) ==
                   minIds .view(-1,1).expand(-1,mySize))).sum() == inputIds.size()[0]*mySize








