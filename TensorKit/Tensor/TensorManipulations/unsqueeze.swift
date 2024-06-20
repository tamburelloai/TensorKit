//
//  unsqueeze.swift
//  TensorKit
//
//  Created by Michael Tamburello on 6/18/24.
//

import Foundation

extension Tensor {
  ///When dim is given, a squeeze operation is done only in the given dimension(s). If input is of shape:
  ///(A×1×B), squeeze(input, 0) leaves the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the shape (A×B).
  public func unsqueeze(_ inputDim: Int) -> Tensor {
    var newTensor = self
    let dim: Int = _adjustForNegativeIndexing(inputDim, offset: 1)
    // Validate the dimension
    precondition(dim >= 0 && dim <= self.shape.count, "Dimension out of range")
    // Create the new shape by inserting 1 at the specified dimension
    newTensor.shape.insert(1, at: dim)
    newTensor.strides = Tensor.calculateStrides(for: newTensor.shape)
    return newTensor
  }
}
