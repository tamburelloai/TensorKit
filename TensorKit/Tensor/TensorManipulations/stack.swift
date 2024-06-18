//
//  stack.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/11/24.
//

import Foundation

/// pre check complimentary to `stack` that ensures list is possible to stack
func validInputToStack<T:TensorData>(_ tensors: [Tensor<T>]) -> Bool {
  guard let firstShape = tensors.first?.shape else { return false }
  return tensors.allSatisfy { $0.shape == firstShape }
}
/// pre check complimentary to `stack` that returns the shape of the stacked tensors
extension Tensor where T: TensorData{
  static func _getStackShape(_ tensors: [Tensor<T>], dim: Int) -> [Int] {
    var resultShape = tensors[0].shape
    resultShape.insert(tensors.count, at: dim)
    return resultShape
  }
}

extension Tensor where T: TensorData {
  static func stack(_ tensors: [Tensor<T>], dim: Int) -> Tensor {
    assert(!tensors.isEmpty)
    assert(validInputToStack(tensors))
    let resultShape: [Int] = _getStackShape(tensors, dim: dim)
    var stackedData: [T] = []
    for i in (0..<resultShape[dim]) {
      stackedData = stackedData + tensors[i].data
    }
    return Tensor(data: stackedData, shape: resultShape)
  }
}
