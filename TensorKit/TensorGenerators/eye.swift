//
//  eye.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

public func eye<T: TensorData>(size: Int, device: DeviceType = .cpu) -> Tensor<T> {
  var shape: [Int] = [size, size] // NxN matrix
  var strides: [Int] = Tensor<T>.calculateStrides(for: shape)
  var data: [T] = Array(repeating: T.zero, count: shape.reduce(1, *))
  for i in 0..<data.count {
    if (i == 0) || (i % (strides[0] + strides[1]) == 0) {
      data[i] = T.one
    }
  }
  return Tensor(data: data, shape: shape, strides: strides, device: device)
}

public func eye<T: TensorData>(_ size: Int, device: DeviceType = .cpu) -> Tensor<T> {
  eye(size: size, device: device)
}
