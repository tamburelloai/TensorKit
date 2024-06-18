//
//  randn.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/7/24.
//

import Foundation


public func randn(_ shape: [Int]) -> Tensor<Float> {
  let totalSize = shape.reduce(1, *)
  let randomData: [Float] = (0..<totalSize).map { _ in Float.sampleFromNormal(mu: 0, sigma: 1) }
  return Tensor(data: randomData, shape: shape)
}

public func randn(_ shape: Int...) -> Tensor<Float> {
    randn(shape)
}


