//
//  ShapeAlignment.swift
//  TensorKit
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation

enum ShapeAlignment {
  case sameShape
  case broadcastAllowed
  case invalidPair
}


func sameShape<T:TensorData&Numeric>(_ tensorA: Tensor<T>, _ tensorB: Tensor<T>) -> Bool {
  return tensorA.shape == tensorB.shape
}

func shapeAlignment<T:TensorData&Numeric>(_ tensorA: Tensor<T>, _ tensorB: Tensor<T>) -> ShapeAlignment {
  if sameShape(tensorA, tensorB) { return .sameShape }
  else if broadcastCompatible(tensorA, tensorB) { return .broadcastAllowed}
  else { return .invalidPair }
}

func deviceLocation<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> DeviceType {
  if lhs.device == rhs.device && rhs.device == DeviceType.mps { return .mps}
  else if lhs.device == rhs.device && rhs.device == DeviceType.cpu { return .cpu}
  else { fatalError("Expect both both tensors to be on the same device: LHS Device: \(lhs.device), RHS Device: \(rhs.device)") }
}
