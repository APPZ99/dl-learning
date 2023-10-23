import torch
import pytorch_cppcuda_learning
import time

def trilinear_interpolation_py(feats, points):

    # 第二维采用 0:1, 是为了保持 tensor 格式
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

# 封装前向和反向函数
class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):
        feat_interp = pytorch_cppcuda_learning.trilinear_interpolation_fw(feats, points)

        # 保存可以进行计算梯度的变量
        ctx.save_for_backward(feats, points)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp):
        # 获取可以计算梯度的变量
        feats, points = ctx.saved_tensors

        dL_dfeats = pytorch_cppcuda_learning.trilinear_interpolation_bw(dL_dfeat_interp.contiguous(), feats, points)

        # ctx 存在两个变量，只要第一个变量的梯度，故第二维需要 None 占位
        return dL_dfeats, None
    

if __name__ == '__main__':
    
    N = 65536; F = 256
    rand = torch.rand(N, 8, F, device='cuda')
    feats = rand.clone().requires_grad_()
    feats2 = rand.clone().requires_grad_()
    points = torch.rand(N, 3, device='cuda') * 2 - 1

    t = time.time()
    # apply 执行前向传播
    out_cuda = Trilinear_interpolation_cuda.apply(feats2, points)
    # 确保所有 cuda 计算完成
    torch.cuda.synchronize()
    print('   cuda fw time', time.time()-t, 's')

    t = time.time()
    out_py = trilinear_interpolation_py(feats, points)
    torch.cuda.synchronize()
    print('pytorch fw time', time.time()-t, 's')

    print('fw all close', torch.allclose(out_py, out_cuda))

    t = time.time()
    loss2 = out_cuda.sum()
    loss2.backward()
    torch.cuda.synchronize()
    print('   cuda bw time', time.time()-t, 's')

    t = time.time()
    loss = out_py.sum()
    loss.backward()
    torch.cuda.synchronize()
    print('pytorch bw time', time.time()-t, 's')

    print('bw all close', torch.allclose(feats.grad, feats2.grad))
