//
//  ScalarTensorArithmetic.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/14/24.
//

import Foundation

extension Tensor<Float>  {
  static func +(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .add) }
  static func +(scalar: Float, tensor: Tensor) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .add) }
  static func -(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .subtract) }
  static func *(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .multiply) }
  static func *(scalar: Float, tensor: Tensor) -> Tensor<Float> { tensor * scalar} // only making scalar mult commutative
  static func /(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .divide) }
  static func /(scalar: Float, tensor: Tensor) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .divide) }

  static private func tensorScalarOp(_ scalar: Float,  _ tensor: Tensor<Float>, _ operation: OperationType) -> Tensor<Float> {
    switch operation {
    case .add: return Tensor(data: tensor.data.map({$0 + scalar}), shape: tensor.shape)
    case .subtract: return Tensor(data: tensor.data.map({$0 - scalar}), shape: tensor.shape)
    case .multiply: return Tensor(data: tensor.data.map({$0 * scalar}), shape: tensor.shape)
    case .divide: return Tensor(data: tensor.data.map({$0 / scalar}), shape: tensor.shape)
    }
  }
}

