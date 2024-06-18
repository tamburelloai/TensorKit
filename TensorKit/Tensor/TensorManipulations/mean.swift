//
//  mean.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/14/24.
//

import Foundation

extension Tensor<Float> {
  func mean() -> Tensor<Float> {
    assert(!data.isEmpty, "Fatal: Attempted to evaluate mean of an empty Tensor")
    let globalMean: Float = data.reduce(Float.zero, +) / Float(data.count)
    return Tensor(data: [globalMean], shape: [])
  }
  
  // Mean along a specific dimension
  func mean(dim: Int, keepDim: Bool = false) -> Tensor<Float> {
    let N: Float = Float(self.shape[dim])
    switch keepDim {
    case true: return (self.reduce(+, dim: dim) / N)
    case false: return (self.reduce(+, dim: dim).squeeze(dim) / N)
    }
  }
}
