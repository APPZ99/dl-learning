import torch
import pytorch_cppcuda_learning

if __name__ == '__main__':
    
    feats = torch.ones(2, device='cuda')
    points = torch.zeros(2, device='cuda')

    out = pytorch_cppcuda_learning.trilinear_interpolation(feats, points)

    print(out)
