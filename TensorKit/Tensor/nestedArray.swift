//
//  nestedArray.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

extension Tensor {
  public func nestedArray() -> Any {
    return createNestedArray(dim: 0, index: [])
  }
  
  private func createNestedArray(dim: Int, index: [Int]) -> Any {
    if dim == shape.count - 1 {
      return (0..<shape[dim]).map { element(at: index + [$0]) }
    } else {
      return (0..<shape[dim]).map { createNestedArray(dim: dim + 1, index: index + [$0]) }
    }
  }
}

