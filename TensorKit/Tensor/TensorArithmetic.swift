//
//  TensorArithmetic.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//


enum OperationType {
  case add
  case subtract
  case multiply
  case divide
}



extension Tensor where T: TensorData & Numeric & FloatingPoint {
  static func +(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> { return performOperation(lhs, rhs, .add) }
  static func -(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> { return performOperation(lhs, rhs, .subtract) }
  static func *(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> { return performOperation(lhs, rhs, .multiply) }
  static func /(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> { return performOperation(lhs, rhs, .divide) }
  
  static private func performOperation(_ lhs: Tensor<T>, _ rhs: Tensor<T>, _ operation: OperationType) -> Tensor<T> {
    switch shapeAlignment(lhs, rhs) {
    case .sameShape: return elementwiseOperation(lhs, rhs, operation: operation)
    case .broadcastAllowed: return broadcastOperation(lhs, rhs, operation)
    case .invalidPair: fatalError("Shapes cannot be broadcast together: \(lhs.shape) and \(rhs.shape)")
    }
  }
  
  static private func elementwiseOperation(_ lhs: Tensor<T>, _ rhs: Tensor<T>, operation: OperationType) -> Tensor<T> {
    switch deviceLocation(lhs, rhs) {
    case .mps: return MPSBackend.shared.elementwiseOperation(lhs, rhs, operation) //  MPSBackend.elementwiseOperation(lhs, rhs, operation)
    case .cpu: return CPUBackend.elementwiseOperation(lhs, rhs, operation)
    }
  }
  
  static private func broadcastOperation(_ lhs: Tensor<T>, _ rhs: Tensor<T>, _ operation: OperationType) -> Tensor<T> {
    switch deviceLocation(lhs, rhs) {
    case .mps: return MPSBackend.shared.broadcastOperation(lhs, rhs, operation)
    case .cpu: return CPUBackend.broadcastOperation(lhs, rhs, operation)
    }
  }
  
  
}
