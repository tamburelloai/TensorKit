//
//  leakyReLU.metal
//  Scorch
//
//  Created by Michael Tamburello on 6/19/24.
//

#include <metal_stdlib>
using namespace metal;


kernel void leakyReLUForward(
                             const device float* input [[buffer(0)]],
                             constant float& alpha [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             uint index [[thread_position_in_grid]])
{
  float x = input[index];
  output[index] = x >= 0 ? x: (alpha * x);
}
kernel void leakyReLUBackward(
                             const device float* input [[buffer(0)]],
                             constant float& alpha [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             uint index [[thread_position_in_grid]])
{
  float x = input[index];
  output[index] = x >= 0 ? 1 : alpha;
}
