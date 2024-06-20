//
//  zerosLike.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

// External function to create a tensor filled with zeros
public func zerosLike<T: TensorData>(_ tensor: Tensor<T>) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  let totalElements = tensor.shape.reduce(1, *)
  let zerosData: [T] = Array(repeating: T.zero, count: totalElements)
  return Tensor(data: zerosData, shape: tensor.shape, device: tensor.device)
}



