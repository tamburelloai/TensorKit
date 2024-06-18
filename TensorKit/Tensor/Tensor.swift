//
//  Tensor.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/6/24.
//

import Foundation
public struct Tensor<T:TensorData> {
  var data: [T]
  var shape: [Int]
  var strides: [Int]
  var device: DeviceType
  
  var ndim: Int { return self.shape.count }
  
  /// Subscript for indexing the tensor with **multiple** dimensions
  /// Example: Getting the (i,j) value - `Tensor[i, j]` => value at index (i, j)
  public subscript(indices: Int...) -> T {
    get {
      let index = calculateIndex(indices: indices)
      return data[index]
    }
    set {
      let index = calculateIndex(indices: indices)
      data[index] = newValue
    }
  }
  
  public func element(at index: [Int]) -> T {
    var flatIndex = 0
    for (i, idx) in index.enumerated() {
      flatIndex += idx * strides[i]
    }
    return data[flatIndex]
  }
  
  private func calculateIndex(indices: [Int]) -> Int {
    assert(indices.count == shape.count, "Index count does not match shape dimensions.")
    return indices.enumerated().reduce(0) { $0 + $1.element * strides[$1.offset] }
  }
  
  func to(_ device: DeviceType) -> Tensor {
    var newTensor = self
    newTensor.device = device
    return newTensor
  }
}

extension Tensor: CustomStringConvertible {
  public var description: String {
    return "\(self.nestedArray())"
  }
}
