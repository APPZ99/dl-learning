#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
){
    // 索引到特定的 thread 线程
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    // 如果不在 block 范围内则 return
    if(n >= feats.size(0) || f >= feats.size(2))
        return;

    // 计算每个坐标的值
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;
    // 计算权重
    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = v * w;
    // 三线性插值
    feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] + b * feats[n][1][f] +
                                   c * feats[n][2][f] + d * feats[n][3][f]) +
                        u       * (a * feats[n][4][f] + b * feats[n][5][f] +
                                   c * feats[n][6][f] + d * feats[n][7][f]);
}


torch::Tensor trilinear_fw_cu(torch::Tensor feats, torch::Tensor points){

    // 获取进行并行计算的两个维度
    const int N = feats.size(0);
    const int F = feats.size(2);

    // 插值后的特征向量 （N, F）
    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());

    // 256个线程数，平均给 N 和 F
    const dim3 threads(16, 16);
    // 计算 block 的大小
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    // 进行计算
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
    ([&] {
        trilinear_fw_kernel<scalar_t> <<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));


    return feat_interp;
}


template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats
){
    // 索引到特定的 thread 线程
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    // 如果不在 block 范围内则 return
    if(n >= feats.size(0) || f >= feats.size(2))
        return;

    // 计算每个坐标的值
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    // 计算权重
    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = v * w;

    // 计算偏微分
    dL_dfeats[n][0][f] = (1 - u) * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1 - u) * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1 - u) * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1 - u) * d * dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u * d * dL_dfeat_interp[n][f];
}



torch::Tensor trilinear_bw_cu(const torch::Tensor dL_dfeat_interp, torch::Tensor feats, torch::Tensor points){

    // 获取进行并行计算的两个维度
    const int N = feats.size(0);
    const int F = feats.size(2);

    // 偏微分的维度 (N, 8, F)
    torch::Tensor dL_dfeats = torch::empty({N, 8, F}, feats.options());

    // 256个线程数，平均给 N 和 F
    const dim3 threads(16, 16);
    // 计算 block 的大小
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu",
    ([&] {
        trilinear_bw_kernel<scalar_t> <<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));


    return dL_dfeats;
}