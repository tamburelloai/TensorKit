//
//  eye.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

public func eye<T: TensorData>(shape: Int) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  var values: [[T]] = []
  var tmp: [T] = []
  for i in 0..<shape {
    tmp = Array(repeating: T.zero, count: shape)
    tmp[i] = T.zero + 1
    values.append(tmp)
  }
  return Tensor(values)
}

public func eye<T: TensorData>(_ shape: Int) -> Tensor<T> where T: Numeric, T: ExpressibleByIntegerLiteral {
  eye(shape: shape)
}
