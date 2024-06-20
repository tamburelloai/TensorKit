//
//  sigmoidBackward.metal
//  Scorch
//
//  Created by Michael Tamburello on 6/19/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void sigmoidBackward(
                            const device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            uint index [[thread_position_in_grid]])
{
    float sigmoidValue = 1.0 / (1.0 + exp(-input[index]));
    output[index] = sigmoidValue * (1.0 - sigmoidValue);
}
