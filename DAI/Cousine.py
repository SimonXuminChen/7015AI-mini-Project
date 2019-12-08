import torch

x=torch.Tensor([[1,2],[3,4],[3,5],[2,3]])
y=torch.Tensor([[1,2],[2,4],[3,4],[2,3]])

result=torch.cosine_similarity(x,y,dim=1)

print(result)
