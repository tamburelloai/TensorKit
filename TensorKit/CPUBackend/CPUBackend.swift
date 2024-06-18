//
//  CPUBackend.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation

class CPUBackend {
  static let shared = CPUBackend() // singleton instance
  
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


/// Matrix-Matrix Operations
extension CPUBackend {
  func dot<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let N: Int = lhs.shape[0]
    let resultShape: [Int] = [1]
    var result: T = zip(lhs.data, rhs.data).map(*).reduce(0, +)
    return Tensor(data: [result], shape: resultShape, device: lhs.device)
  }
  
  func matMul<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let (M, K) = (lhs.shape[0], lhs.shape[1])
    let (N, P) = (rhs.shape[0], rhs.shape[1])
    let resultShape: [Int] = [M, P]
    var result: [T] = Array(repeating: T.zero, count: resultShape.reduce(1,*))
    for i in 0..<M {
      for j in 0..<P {
        var sum: T = 0
        for k in 0..<K {
          let lhsIndex = i * K + k
          let rhsIndex = k * P + j
          if lhsIndex >= lhs.data.count {
            print("LHS index out of bounds: \(lhsIndex)")
          }
          if rhsIndex >= rhs.data.count {
            print("RHS index out of bounds: \(rhsIndex)")
          }
          sum += lhs.data[lhsIndex] * rhs.data[rhsIndex]
        }
        result[i * P + j] = sum
      }
    }
    return Tensor(data: result, shape: resultShape, device: lhs.device)
  }
  
  
  
  //  func batchMatMul(lhs: Tensor, rhs: Tensor) -> Tensor? {
  //    let batchSize = lhs.shape[0]
  //    let N = lhs.shape[1]
  //    let M = lhs.shape[2]
  //    let P = rhs.shape[2]
  //    let outShape = [batchSize, N, P]
  //
  //    // Implementation for batch matrix product
  //    var result = [Float](repeating: 0, count: batchSize * N * P)
  //
  //    for b in 0..<batchSize {
  //      for i in 0..<N {
  //        for j in 0..<P {
  //          var sum: Float = 0
  //          for k in 0..<M {
  //            sum += lhs.data[b * N * M + i * M + k] * rhs.data[b * M * P + k * P + j]
  //          }
  //          result[b * N * P + i * P + j] = sum
  //        }
  //      }
  //    }
  //
  //    return Tensor(values: result, shape: outShape, device: lhs.device)
  // }
}
