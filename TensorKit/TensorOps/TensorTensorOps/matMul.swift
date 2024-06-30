//
//  matMul.swift
//  TensorKit
//
//  Created by Michael Tamburello on 6/18/24.
//

import Foundation

extension Tensor where T: TensorData & Numeric {
  
  public static func matMul(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.ndim == 2 && rhs.ndim == 2, "fatal: Tensor.matMul requires 2D Tensors")
    assert(lhs.shape[1] == rhs.shape[0], "fatal: Tensor.matMul requires matching inner dims, got \(lhs.shape) and \(rhs.shape)")
    if lhs.device == .cpu && rhs.device == .cpu { return CPUBackend.shared.matMul(lhs, rhs) }
    else if lhs.device == .mps && rhs.device == .mps { return MPSBackend.shared.matMul(lhs, rhs) }
    else { fatalError() }
  }
  
  public func matMul(_ rhs: Tensor<T>) -> Tensor<T> {
    assert(self.ndim == 2 && rhs.ndim == 2, "fatal: Tensor.matMul requires 2D Tensors")
    assert(self.shape[1] == rhs.shape[0], "fatal: Tensor.matMul requires matching inner dims, got \(self.shape) and \(rhs.shape)")
    if self.device == .cpu && rhs.device == .cpu { return CPUBackend.shared.matMul(self, rhs) }
    else if self.device == .mps && rhs.device == .mps { return MPSBackend.shared.matMul(self, rhs) }
    else { fatalError() }
  }
  
  
}
