//
//  ScalarTensorArithmetic.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/14/24.
//

import Foundation

extension Tensor<Float>  {
  public static func +(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .add) }
  public static func +(scalar: Float, tensor: Tensor) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .add) }
  public static func -(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .subtract) }
  public static func *(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .multiply) }
  public static func *(scalar: Float, tensor: Tensor) -> Tensor<Float> { tensor * scalar} // only making scalar mult commutative
  public static func /(tensor: Tensor, scalar: Float) -> Tensor<Float> { return tensorScalarOp(scalar, tensor, .divide) }
  public static func /(scalar: Float, tensor: Tensor) -> Tensor<Float> { tensor / scalar} // only making scalar mult commutative
  
  
  public static func +(tensor: Tensor, scalar: Double) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .add) }
  public static func +(scalar: Double, tensor: Tensor) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .add) }
  public static func -(tensor: Tensor, scalar: Double) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .subtract) }
  public static func *(tensor: Tensor, scalar: Double) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .multiply) }
  public static func *(scalar: Double, tensor: Tensor) -> Tensor<Float> { tensor * Float(scalar)} // only making scalar mult commutative
  public static func /(tensor: Tensor, scalar: Double) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .divide) }
  public static func /(scalar: Double, tensor: Tensor) -> Tensor<Float> { tensor / Float(scalar)} // only making scalar mult commutative
  
  
  public static func +(tensor: Tensor, scalar: Int) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .add) }
  public static func +(scalar: Int, tensor: Tensor) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .add) }
  public static func -(tensor: Tensor, scalar: Int) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .subtract) }
  public static func *(tensor: Tensor, scalar: Int) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .multiply) }
  public static func *(scalar: Int, tensor: Tensor) -> Tensor<Float> { tensor * Float(scalar)} // only making scalar mult commutative
  public static func /(tensor: Tensor, scalar: Int) -> Tensor<Float> { return tensorScalarOp(Float(scalar), tensor, .divide) }
  public static func /(scalar: Int, tensor: Tensor) -> Tensor<Float> { tensor / Float(scalar)} // only making scalar mult commutative
  
  private static func tensorScalarOp(_ scalar: Float,  _ tensor: Tensor<Float>, _ operation: OperationType) -> Tensor<Float> {
    switch operation {
    case .add: return Tensor(data: tensor.data.map({$0 + scalar}), shape: tensor.shape)
    case .subtract: return Tensor(data: tensor.data.map({$0 - scalar}), shape: tensor.shape)
    case .multiply: return Tensor(data: tensor.data.map({$0 * scalar}), shape: tensor.shape)
    case .divide: return Tensor(data: tensor.data.map({$0 / scalar}), shape: tensor.shape)
    }
  }
}

