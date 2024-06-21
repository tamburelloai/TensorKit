//
//  dotProduct.metal
//  TensorKit
//
//  Created by Michael Tamburello on 6/18/24.
//

#include <metal_stdlib>
using namespace metal;



// This function calculates the dot product of two vectors
kernel void dot_Float(
                       device const float* vectorA [[ buffer(0) ]],  // Input vector A
                       device const float* vectorB [[ buffer(1) ]],  // Input vector B
                       device float* result [[ buffer(2) ]],         // Output result
                       constant int& vectorLength [[ buffer(3) ]],   // Length of the vectors
                       uint id [[ thread_position_in_grid ]])        // Thread identifier
{
  if (id >= vectorLength) return; // Ensures we do not access out of bounds memory
  
  // Using a temporary variable to accumulate the dot product
  float dotProduct = 0.0;
  for (int i = 0; i < vectorLength; ++i) {
    dotProduct += vectorA[i] * vectorB[i];
  }
  
  // Only the first thread writes the result
  if (id == 0) {
    *result = dotProduct;
  }
}

// This function calculates the dot product of two vectors
kernel void dot_Int(
                       device const int* vectorA [[ buffer(0) ]],  // Input vector A
                       device const int* vectorB [[ buffer(1) ]],  // Input vector B
                       device int* result [[ buffer(2) ]],         // Output result
                       constant int& vectorLength [[ buffer(3) ]],   // Length of the vectors
                       uint id [[ thread_position_in_grid ]])        // Thread identifier
{
  if (id >= vectorLength) return; // Ensures we do not access out of bounds memory
  
  // Using a temporary variable to accumulate the dot product
  int dotProduct = 0;
  for (int i = 0; i < vectorLength; ++i) {
    dotProduct += vectorA[i] * vectorB[i];
  }
  
  // Only the first thread writes the result
  if (id == 0) {
    *result = dotProduct;
  }
}

