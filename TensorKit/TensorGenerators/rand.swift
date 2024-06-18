//
//  rand.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/7/24.
//

import Foundation

public func rand(_ shape: [Int]) -> Tensor<Float> {
  let totalSize = shape.reduce(1, *)
  let randomData: [Float] = (0..<totalSize).map { _ in Float.sampleFromUniform(a: 0, b: 1) }
  return Tensor(data: randomData, shape: shape)
}

public func rand(_ shape: Int...) -> Tensor<Float> {
  rand(shape)
}


