//
//  elementwiseDivision.metal
//  Scorch
//
//  Created by Michael Tamburello on 6/12/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void elementwiseDivision(
  const device float* tensorA [[buffer(0)]],
  const device float* tensorB [[buffer(1)]],
  device float* result [[buffer(2)]],
  uint index [[thread_position_in_grid]])
{
  result[index] = tensorA[index] / tensorB[index];
}


