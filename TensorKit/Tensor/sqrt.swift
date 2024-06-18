//
//  sqrt.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/15/24.
//

import Foundation

extension Tensor<Float> {
  static func sqrt(_ inputTensor: Tensor<Float>) -> Tensor<Float> {
    return Tensor<Float>(
      data: inputTensor.data.map {($0.squareRoot())},
      shape: inputTensor.shape
    )
  }
}

