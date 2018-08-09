import torch

mySeed = 17
torch.manual_seed(mySeed)
myLow = 0
myHigh = 5
mySize = 10

num_sampled = 3
x=(torch.randint(low=myLow,high=myHigh,size=(mySize,3))
   .sort(dim=1))[0]
minIds = x[:,0]
inputIds = x[:,1]
maxIds = x[:,2]

# create the rescaling width; need to add 1.0 as sampler
# in [0,1)
width = maxIds-minIds+1.0
sampled0 = torch.rand(size=(minIds.size()[0],num_sampled))
# rescale the sampled0 by width
sampled1 = width.view(-1,1)*sampled0
# add the minimum element so we fall in the sampled items
sampled2 = minIds.view(-1,1)+sampled1
# floor to integer
sampled3 = sampled2.floor()
# increase when there is a collision
sampled4 = torch.where(sampled3 == inputIds.view(-1,1),sampled3+1.0,sampled3)
# decrease again when we go above the maximum allowed value
sampled5 = torch.where(sampled4 > maxIds.view(-1,1),minIds.view(-1,1).expand(-1,num_sampled),
                       sampled4)
print(x)
print(sampled3)
print(sampled4)
print(sampled5)
print(x)






