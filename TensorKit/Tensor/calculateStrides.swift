//
//  calculateStrides.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

extension Tensor {
  // Calculate strides for efficient indexing
  static func calculateStrides(for shape: [Int]) -> [Int] {
    var strides = Array(repeating: 1, count: shape.count)
    for i in stride(from: shape.count - 2, through: 0, by: -1) {
      strides[i] = strides[i + 1] * shape[i + 1]
    }
    return strides
  }
}
