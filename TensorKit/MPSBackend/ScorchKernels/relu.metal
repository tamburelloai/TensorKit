//
//  relu.metal
//  Scorch
//
//  Created by Michael Tamburello on 6/19/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void reluForward(
                        const device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        uint index [[thread_position_in_grid]])
{
  float inValue = input[index];
  output[index] = inValue > 0 ? inValue : 0;
}

kernel void reluBackward(
                         const device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         uint index [[thread_position_in_grid]])
{
  float inValue = input[index];
  output[index] = inValue > 0 ? 1 : 0;
}
