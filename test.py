import torch
import pytorch_cppcuda_learning

def trilinear_interpolation_py(feats, points):

    u = (points[:, 0:1] + 1) / 2
    v = (points[:, 1:2] + 1) / 2
    w = (points[:, 2:3] + 1) / 2

    a = (1 - v) * (1 - w)
    b = (1 - v) * w
    c = v * (1 - w)
    d = v * w

    feats_interp = (1 - u) * (a * feats[:, 0] + b * feats[:, 1] +
                              c * feats[:, 2] + d * feats[:, 3]) + \
                    u      * (a * feats[:, 4] + b * feats[:, 5] +
                              c * feats[:, 6] + d * feats[:, 7])
    
    return feats_interp


if __name__ == '__main__':
    
    N = 65536; F = 256
    feats = torch.rand(N, 8, F, device = 'cuda').requires_grad_()
    points = torch.rand(N, 3, device = 'cuda')

    print(points[0, 0], points[0, 1], points[0, 2])
    print(points[0, 0:1], points[0, 1:2], points[0, 2:3])
    
    out_cuda = pytorch_cppcuda_learning.trilinear_interpolation(feats, points)

    out_py = trilinear_interpolation_py(feats, points)

    print(torch.allclose(out_cuda, out_py))

