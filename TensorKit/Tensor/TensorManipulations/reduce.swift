//
//  reduce.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/13/24.
//

import Foundation

extension Tensor where T: TensorData & Numeric & FloatingPoint {
  func reduce(_ op: (T, T) -> T, dim: Int) -> Tensor<T> {
      // Set the output shape by reducing the dimension we'll combine into 1
      var outputShape = self.shape
      outputShape[dim] = 1
      
      // Get length of input data array and output data array
      let numInputElements = self.shape.reduce(1, *)
      let numOutputElements = outputShape.reduce(1, *)
      // Create empty array for output
      var output: [T] = Array(repeating: T.zero, count: numOutputElements)
      
      // Calculate the strides for all dimensions
      var strides = Array(repeating: 1, count: self.shape.count)
      for i in (1..<self.shape.count).reversed() {
        strides[i-1] = strides[i] * self.shape[i]
      }
      
      // Iterate over all elements of the original tensor
      for idx in 0..<numInputElements {
        var multidimIndex = indexToMultiDim(idx, self.shape)
        multidimIndex[dim] = 0 // Reduce along the specified dimension
        let resultIdx = multiDimToIndex(multidimIndex, outputShape)
        output[resultIdx] = op(output[resultIdx], self.data[idx])
      }
      
      return Tensor(data: output, shape: outputShape, device: self.device)
    }
}
