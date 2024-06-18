//
//  matMul.metal
//  TensorKit
//
//  Created by Michael Tamburello on 6/18/24.
//

#include <metal_stdlib>
using namespace metal;

/// Matrix multiplication between two tensors
/// `tensorA` and `tensorB` are passed to buffers (memory allocations on the GPU), and then
/// a thread is assigned to each of them. The results are stored in `tensorC`, which has its own buffer
/// and will be read back to the CPU to get our summed result.
kernel void matMul(
                          const device float* tensorA [[buffer(0)]],
                          const device float* tensorB [[buffer(1)]],
                          device float* tensorC [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& K [[buffer(4)]],
                          constant uint& N [[buffer(5)]],
                          uint id [[thread_position_in_grid]])
{
  uint row = id / N;
  uint col = id % N;
  float sum = 0.0;
  for (uint k = 0; k < K; ++k) {
    sum += tensorA[row*K+k] * tensorB[k*N+col];
  }
  tensorC[row*N+col] = sum;
}

