//
//  TensorBroadcasting.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/11/24.
//

import Foundation

enum TensorError: Error {
    case incompatibleShapes
}

func initZeros(_ shape: [Int]) -> [Float] {
  return Array(repeating: 0, count: shape.reduce(1, *))
}

func calculateBroadcastShape(_ shapeA: [Int], _ shapeB: [Int]) -> [Int] {
  let maxDims: Int = max(shapeA.count, shapeB.count)
  let paddedShapeA: [Int] = padShapeWithOnes(shape: shapeA, maxDims: maxDims)
  let paddedShapeB: [Int] = padShapeWithOnes(shape: shapeB, maxDims: maxDims)
  var resultShape: [Int] = Array(repeating: 0, count: maxDims)
  for i in (0..<maxDims) {
    resultShape[i] = max(paddedShapeA[i], paddedShapeB[i])
  }
  return resultShape
}

func indexToMultiDim(_ idx: Int, _ shape: [Int]) -> [Int] {
  var index = idx
  var multiDimIndex: [Int] = Array(repeating: 0, count: shape.count)
  let numDims: Int = shape.count
  for i in stride(from: numDims-1, through: 0, by: -1) {
    multiDimIndex[i] = index % shape[i]
    //    index = Int(index / shape[i])
    index = index / shape[i]
    
  }
  return multiDimIndex
}

func multiDimToIndex(_ multiDimIndex: [Int], _ shape: [Int]) -> Int {
  var index = 0
  for i in (0..<shape.count) {
    index = index * shape[i] + multiDimIndex[i]
  }
  return index
}

func padShapeWithOnes(shape: [Int], maxDims: Int) -> [Int] {
  let paddingAmount: Int = maxDims - shape.count
  let paddingDims: [Int] = Array(repeating: 1, count: paddingAmount)
  return paddingDims + shape
}

func productOfDimensions(_ dims: [Int]) -> Int {
  return dims.reduce(1, *)
}

func broadcastCompatible<T:TensorData>(_ tensorA: Tensor<T>, _ tensorB: Tensor<T>) -> Bool {
  let N: Int = min(tensorA.shape.count, tensorB.shape.count)
  let shapeA: [Int] = tensorA.shape.reversed()
  let shapeB: [Int] = tensorB.shape.reversed()
  for i in (0..<N) {
    if (shapeA[i] == shapeB[i] || shapeA[i] == 1 || shapeB[i] == 1) {
      continue
    } else { return false }
  }
  return true
}



extension Tensor where T: TensorData & Numeric & FloatingPoint {
  static func _broadcastOperation(_ tensorA: Tensor<T>, _ tensorB: Tensor<T>, operation: (T, T) -> T) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(tensorA.shape, tensorB.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: tensorA.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: tensorB.shape, maxDims: resultShape.count)
    
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
      resultFlatArray[idx] = operation(tensorA.data[indexA], tensorB.data[indexB])
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
}
