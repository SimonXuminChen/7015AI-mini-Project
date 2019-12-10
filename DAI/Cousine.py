import torch

x=torch.Tensor([[1,2],[3,4],[3,5],[2,3]])
y=torch.Tensor([[1,2],[2,4],[3,4],[2,3]])
a=torch.Tensor([[1,2,3,4,5,6],
               [2,3,4,5,6,7],
                [3,4,5,6,7,8]])
z=a[:,2]

# result=torch.cosine_similarity(x,y,dim=1)


print(a)
print(x)
print(y)
print(z)
# print(result)
