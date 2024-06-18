//
//  variance.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/14/24.
//

import Foundation

extension Tensor<Float> {
  
  // Mean along a specific dimension
  func variance(dims: [Int] = [], correction: Int = 1, keepDim: Bool = false) -> Tensor<Float> {
    switch dims.count {
    case 0: return _globalVariance()
    case 1: return _singleDimVariance(dim: dims.first!, keepDim: keepDim)
    default: return _multiDimVariance(dims: dims)
    }
  }
  
  func variance(dim: Int, correction: Int = 1, keepDim: Bool = false) -> Tensor<Float> {
    return _singleDimVariance(dim: dim, correction: correction, keepDim: keepDim)
  }
  
  private func _globalVariance(correction: Int = 1) -> Tensor<Float> {
    assert(!data.isEmpty, "Fatal: Attempted to evaluate mean of an empty Tensor")
    let N: Int = self.data.count
    let scaleFactor: Float = (1/Float(max(0, N-correction)))
    let globalMean: Float = self.mean().item()
    let globalMeanReduced: Tensor<Float> = self - globalMean
    let globalMeanReducedSquared: Tensor<Float> = globalMeanReduced * globalMeanReduced
    let globalMeanReducedSquaredSum: Float = globalMeanReducedSquared.data.reduce(Float.zero, +)
    let globalVariance = scaleFactor * globalMeanReducedSquaredSum
    return Tensor(data: [globalVariance], shape: [])
  }
  
  private func _singleDimVariance(dim: Int, correction: Int = 1, keepDim: Bool = false) -> Tensor<Float> {
    assert(self.shape.count > dim)
    let N: Int = self.shape[dim]
    let scaleFactor: Float = (1/Float(max(0, N-correction)))
    var dimMeans: Tensor<Float> = self.mean(dim: dim, keepDim: true)
    var dimDifferences: Tensor<Float> = self - dimMeans
    let meanDeltaSquared: Tensor<Float> = dimDifferences*dimDifferences //TODO: power function integration
    let result: Tensor<Float> = scaleFactor * meanDeltaSquared.reduce(+, dim: dim)
    switch keepDim {
    case true: return result
    case false: return result.squeeze(dim)
    }
  }
  
  private func _multiDimVariance(dims: [Int]) -> Tensor<Float> {
    return zeros(1)
  }
  
  
  
}
