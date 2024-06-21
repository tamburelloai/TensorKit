//
//  ones.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

public func ones<T: TensorData>(shape: [Int], device: DeviceType = .cpu) -> Tensor<T> {
  let totalElements = shape.reduce(1, *)
  let onesData: [T] = Array(repeating: T.one, count: totalElements)
  return Tensor(data: onesData, shape: shape)
}

public func ones<T: TensorData>(_ shape: Int..., device: DeviceType = .cpu) -> Tensor<T> {
  ones(shape: shape)
}
