//
//  batchMatMul.metal
//  TensorKit
//
//  Created by Michael Tamburello on 6/18/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void batchMatrixProduct_Float(
                               device float* lhs [[buffer(0)]],
                               device float* rhs [[buffer(1)]],
                               constant uint& batchSize [[buffer(2)]],
                               constant uint& n [[buffer(3)]],
                               constant uint& m [[buffer(4)]],
                               constant uint& p [[buffer(5)]],
                               uint id [[thread_position_in_grid]],
                               device float* result [[buffer(6)]])
{
  uint batch = id / (n * p);
  uint row = (id % (n * p)) / p;
  uint col = id % p;
  
  if (batch < batchSize) {
    float sum = 0.0;
    for (uint k = 0; k < m; ++k) {
      sum += lhs[batch * n * m + row * m + k] * rhs[batch * m * p + k * p + col];
    }
    result[batch * n * p + row * p + col] = sum;
  }
}

kernel void batchMatrixProduct_Int(
                               device int* lhs [[buffer(0)]],
                               device int* rhs [[buffer(1)]],
                               constant uint& batchSize [[buffer(2)]],
                               constant uint& n [[buffer(3)]],
                               constant uint& m [[buffer(4)]],
                               constant uint& p [[buffer(5)]],
                               uint id [[thread_position_in_grid]],
                               device int* result [[buffer(6)]])
{
  uint batch = id / (n * p);
  uint row = (id % (n * p)) / p;
  uint col = id % p;
  
  if (batch < batchSize) {
    int sum = 0;
    for (uint k = 0; k < m; ++k) {
      sum += lhs[batch * n * m + row * m + k] * rhs[batch * m * p + k * p + col];
    }
    result[batch * n * p + row * p + col] = sum;
  }
}
