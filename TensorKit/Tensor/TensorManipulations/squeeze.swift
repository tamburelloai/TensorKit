//
//  squeeze.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/14/24.
//

import Foundation


extension Tensor {
  ///Returns a tensor with all specified dimensions of input of size 1 removed.
  ///For example, if input is of shape:
  ///(A×1×B×C×1×D) then the input.squeeze() will be of shape: (A×B×C×D).
  public func squeeze() -> Tensor {
    var newTensor = self
    newTensor.shape = self.shape.filter { $0 != 1 }
    newTensor.strides = Tensor.calculateStrides(for: newTensor.shape) //TODO: clean this up to happen any time changes to shape are made in a Tensor
    return newTensor
  }
  
  ///When dim is given, a squeeze operation is done only in the given dimension(s). If input is of shape:
  ///(A×1×B), squeeze(input, 0) leaves the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the shape (A×B).
  public func squeeze(_ inputDim: Int) -> Tensor {
    let dim: Int = _adjustForNegativeIndexing(inputDim)
    assert(self.shape.count > dim)
    switch (self.shape[dim] == 1) {
    case true: return _squeezeAtDim(dim)
    case false: return self
    }
  }
  
  func _adjustForNegativeIndexing(_ dim: Int, offset: Int = 0) -> Int {
    return dim < 0 ? dim + self.shape.count + offset : dim
  }
  ///When dim is given, a squeeze operation is done only in the given dimensions. If input is of shape:
  ///(A×1×B), squeeze(input, 0) leaves the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the shape (A×B).
  public func squeeze(dims: [Int]) -> Tensor {
    var dimsToSqueeze: [Int] = dims
    var newTensor: Tensor? = nil
    while (!dimsToSqueeze.isEmpty) {
      newTensor = squeeze(dimsToSqueeze.popLast()!)
    }
    guard let result = newTensor else { fatalError() }
    return result
  }
  
  private func _squeezeAtDim(_ dim: Int) -> Tensor {
    var newTensor = self
    var newShape: [Int] = []
    for (idx, element) in self.shape.enumerated() {
      if idx != dim {
        newShape.append(element)
      }
    }
    newTensor.shape = newShape
    newTensor.strides = Tensor.calculateStrides(for: newTensor.shape)
    return newTensor
  }
  
  public mutating func squeeze(_ inputDim: Int, inplace: Bool) {
    let dim: Int = _adjustForNegativeIndexing(inputDim)
    switch (self.shape[dim] == 1) {
    case false: return
    default:
      var newShape: [Int] = []
      for (idx, element) in self.shape.enumerated() {
        if idx != dim {
          newShape.append(element)
        }
      }
      self.shape = newShape
      self.strides = Tensor.calculateStrides(for: self.shape)
    }
  }
}
