//
//  transpose.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

extension Tensor {
  public mutating func transpose(_ dim0: Int, _ dim1: Int) {
    assert(dim0 < shape.count && dim1 < shape.count, "Dimension indices are out of bounds.")
    shape.swapAt(dim0, dim1)   // Swap the shape dimensions
    strides.swapAt(dim0, dim1) // Swap the corresponding strides
  }
  
  public func transpose(dim0: Int, dim1: Int) -> Tensor {
    assert(dim0 < shape.count && dim1 < shape.count, "Dimension indices are out of bounds.")
    var newShape = self.shape.map {$0}   // Swap the shape dimensions
    var newStrides = self.strides.map {$0}   // Swap the shape dimensions
    newShape.swapAt(dim0, dim1)
    newStrides.swapAt(dim0, dim1)
    return Tensor(data: self.data, shape: newShape, strides: newStrides, device: self.device)
  }
}
