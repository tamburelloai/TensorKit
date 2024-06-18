//
//  CPUBackend.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation

class CPUBackend {
  static func elementwiseOperation<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>, _ operation: OperationType) -> Tensor<T> {
    switch operation {
    case .add: return Tensor(data: zip(lhs.data, rhs.data).map(+), shape: lhs.shape)
    case .subtract: return Tensor(data: zip(lhs.data, rhs.data).map(-), shape: lhs.shape)
    case .multiply: return Tensor(data: zip(lhs.data, rhs.data).map(*), shape: lhs.shape)
    case .divide: return Tensor(data: zip(lhs.data, rhs.data).map(/), shape: lhs.shape)
    }
  }
  
  static func broadcastOperation<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>, _ operation: OperationType) -> Tensor<T> {
    switch operation {
    case .add: return self._broadcastAdd(lhs, rhs)
    case .subtract: return self._broadcastSubtract(lhs, rhs)
    case .multiply: return self._broadcastMultiply(lhs, rhs)
    case .divide: return self._broadcastDivide(lhs, rhs)
    }
  }
  
  static func _broadcastAdd<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] + rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
  
  static func _broadcastSubtract<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] - rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
  
  static func _broadcastMultiply<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] * rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
  
  static func _broadcastDivide<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] / rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
}
