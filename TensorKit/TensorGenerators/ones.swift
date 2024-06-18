//
//  ones.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

public func ones<T: TensorData>(shape: [Int]) -> Tensor<T> where T: Numeric {
  let totalElements = shape.reduce(1, *)
  let onesData: [T] = Array(repeating: T.zero + 1, count: totalElements)
  return Tensor(data: onesData, shape: shape)
}

func ones<T: TensorData>(_ shape: Int...) -> Tensor<T> where T: Numeric {
  ones(shape: shape)
}
