//
//  concat.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/11/24.
//

import Foundation

/// pre check complimentary to `stack` that ensures list is possible to stack
func validInputToConcat<T:TensorData>(_ tensors: [Tensor<T>], dim: Int) -> Bool {
  let firstShape = tensors[0].shape
  return tensors.allSatisfy { tensor in
    tensor.shape.indices.allSatisfy { index in
      index == dim || tensor.shape[index] == firstShape[index]
    }
  }
}


/// pre check complimentary to `stack` that returns the shape of the stacked tensors
extension Tensor where T: TensorData{
  static func _getConcatShape(_ tensors: [Tensor<T>], dim: Int) -> [Int] {
    var shape = tensors[0].shape
    shape[dim] = tensors.map { $0.shape[dim] }.reduce(0, +)
    return shape
  }
}


extension Tensor where T: TensorData & Numeric {
  static func concat(_ tensors: [Tensor<T>], dim: Int) -> Tensor {
    assert(!tensors.isEmpty)
    assert(validInputToConcat(tensors, dim: dim))
    let resultShape: [Int] = Tensor._getConcatShape(tensors, dim: dim)
    var concatenatedTensor: Tensor = zeros(shape: resultShape)
    var dimensionOffset: Int = 0
    for i in (0..<tensors.count) {
      var tensor: Tensor = tensors[i]
      for index in (0..<tensor.data.count) {
        var offsetIndices: [Int] = indexToMultiDim(index, tensor.shape)
        offsetIndices[dim] += dimensionOffset
        let offsetIndex = multiDimToIndex(offsetIndices, concatenatedTensor.shape)
        concatenatedTensor.data[offsetIndex] = tensor.data[index]
      }
      dimensionOffset += tensor.shape[dim]
    }
    return concatenatedTensor
  }
}



