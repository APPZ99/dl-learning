import torch
import pytorch_cppcuda_learning

feats = torch.ones(2)
point = torch.zeros(2)

out = pytorch_cppcuda_learning.trilinear_interpolation(feats, point)

print(out)