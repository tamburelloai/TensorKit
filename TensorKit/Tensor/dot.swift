//
//  dot.swift
//  TensorKit
//
//  Created by Michael Tamburello on 6/18/24.
//

import Foundation

private func isRowTensor<T:TensorData>(_ tensor: Tensor<T>) -> Bool {
  return (tensor.ndim == 2 && tensor.shape[0] == 1)
}

private func isColTensor<T:TensorData>(_ tensor: Tensor<T>) -> Bool {
  return (tensor.ndim == 2 && tensor.shape[1] == 1)
}

extension Tensor where T: TensorData & Numeric {
  static func dot(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.device == rhs.device, "tensors must be on same device")
    assert(lhs.ndim == rhs.ndim, "tensors must have same dimensions")
    switch lhs.device {
    case .cpu:
      if (lhs.ndim == 1) {
        let result: Tensor<T> = CPUBackend.shared.matMul(lhs.unsqueeze(0), rhs.unsqueeze(-1))
        return result.squeeze(0)
      } else if (lhs.ndim == 2) {
        let result: Tensor<T> = CPUBackend.shared.matMul(lhs, rhs)
        return result
      } else {
        fatalError()
      }
    case .mps:
      if (lhs.ndim == 1) {
        let result: Tensor<T> = MPSBackend.shared.matMul(lhs.unsqueeze(0), rhs.unsqueeze(-1))
        return result.squeeze(0)
      } else if (lhs.ndim == 2) {
        assert(isRowTensor(lhs) && isColTensor(rhs))
        let result: Tensor<T> = MPSBackend.shared.matMul(lhs, rhs)
        return result
      } else {
        fatalError()
      }
    }
  }
  
  func dot(_ rhs: Tensor<T>) -> Tensor<T> {
    return Tensor.dot(self, rhs)
  }
  
  
}
