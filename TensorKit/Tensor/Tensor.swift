//
//  Tensor.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/6/24.
//



import Foundation
public struct Tensor<T:TensorData> {
  public var data: [T]
  public var shape: [Int]
  public var strides: [Int]
  public var device: DeviceType
  
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
  
  public func to(_ device: DeviceType) -> Tensor {
    var newTensor = self
    newTensor.device = device
    return newTensor
  }
  
  //  TODO: fix this T->U Mapping issue on first line may have to create Type.init() for Int, Bool, and Float
  //  public func astype<U: TensorData>(_ type: U.Type) -> Tensor<U> {
  //  let convertedData = self.data.compactMap { U($0) ?? 0 }
  //  if convertedData.count != data.count { fatalError() }
  //  return Tensor<U>(data: convertedData, shape: self.shape, device: self.device)
  //}
  
  // TODO: fix or extent the types to call respective powf powd , etc.
  func pow<U:Numeric>(_ tensor: Tensor<U>, _ exponent: U) -> Tensor<U> {ones(1)}
}

extension Tensor: CustomStringConvertible {
  public var description: String {
    return "\(self.nestedArray())"
  }
}
