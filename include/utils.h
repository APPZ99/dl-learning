#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// 前向传播函数声明
torch::Tensor trilinear_fw_cu(
    const torch::Tensor feats,
    const torch::Tensor points
);

// 反向传播函数声明
torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
);