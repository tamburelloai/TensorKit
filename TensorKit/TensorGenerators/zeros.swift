//
//  zeros.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/7/24.
//

import Foundation

// External function to create a tensor filled with zeros
public func zeros<T: TensorData>(_ shape: [Int]) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  let totalElements = shape.reduce(1, *)
  let zerosData: [T] = Array(repeating: T.zero, count: totalElements)
  return Tensor(data: zerosData, shape: shape)
}

public func zeros<T: TensorData>(shape: [Int]) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  let totalElements = shape.reduce(1, *)
  let zerosData: [T] = Array(repeating: T.zero, count: totalElements)
  return Tensor(data: zerosData, shape: shape)
}

public func zeros<T: TensorData>(like tensor: Tensor<T>, device: DeviceType) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  let totalElements = tensor.shape.reduce(1, *)
  let zerosData: [T] = Array(repeating: T.zero, count: totalElements)
  return Tensor(data: zerosData, shape: tensor.shape, device: tensor.device)
}


public func zeros<T: TensorData>(_ shape: Int...) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  zeros(shape: shape)
}


public func zeros<T: TensorData>(like tensor: Tensor<T>) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  zeros(shape: tensor.shape)
}



