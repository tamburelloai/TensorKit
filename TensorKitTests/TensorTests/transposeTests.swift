//
//  transposeTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//

import XCTest
@testable import TensorKit

extension TensorTests {
  /// `Placeholder for test. Already heavily tested though in TensorInitTests`
  func test2DTransposeVector() {
    let N: Int = Int.random(in: 0...100)
    let values: [[Float]] = [Array(repeating: 1, count: N).map { _ in Float.sampleFromNormal(mu: 0, sigma: 1)}]
    let transposedValues: [[Float]] = values[0].map { [$0]}
    let rowTensor: Tensor<Float> = Tensor(values)
    let columnTensor: Tensor<Float> = Tensor(transposedValues)
    let rowToColumn: Tensor<Float> = rowTensor.transpose(dim0: 0, dim1: 1)
    let columnToRow = columnTensor.transpose(dim0: 0, dim1: 1)
    XCTAssertEqual(rowTensor.data, columnTensor.data)
    XCTAssertEqual(rowTensor.data, rowToColumn.data)
    XCTAssertEqual(columnTensor.data, columnToRow.data)
    XCTAssertEqual(rowTensor.nestedArray as! [[Float]], values)
    XCTAssertEqual(columnTensor.nestedArray as! [[Float]], transposedValues)
    XCTAssertEqual(rowToColumn.nestedArray as! [[Float]], transposedValues)
    XCTAssertEqual(columnToRow.nestedArray as! [[Float]], values)
  }
  
  func test2DTransposeVectorMutating() {
    let N: Int = Int.random(in: 0...100)
    let values: [[Float]] = [Array(repeating: 1, count: N).map { _ in Float.sampleFromNormal(mu: 0, sigma: 1)}]
    let transposedValues: [[Float]] = values[0].map { [$0]}
    let rowTensor: Tensor<Float> = Tensor(values)
    let columnTensor: Tensor<Float> = Tensor(transposedValues)
    XCTAssertEqual(rowTensor.transpose(dim0: 0, dim1: 1).data, columnTensor.data)
    XCTAssertEqual(columnTensor.transpose(dim0: 0, dim1: 1).nestedArray as! [[Float]], values)
    XCTAssertEqual(rowTensor.transpose(dim0: 0, dim1: 1).nestedArray as! [[Float]],  rowTensor.transpose(dim0: 1, dim1: 0).nestedArray as! [[Float]])
  }
  
  func test3DTranspose() {
    func generate3DArray(dim1: Int, dim2: Int, dim3: Int) -> [[[Float]]] {
      return (0..<dim1).map { _ in
        (0..<dim2).map { _ in
          Array(repeating: 1, count: dim3).map { _ in Float.random(in: -1...1) }
        }
      }
    }
    let dim1 = Int.random(in: 1...10)
    let dim2 = Int.random(in: 1...10)
    let dim3 = Int.random(in: 1...10)
    let values = generate3DArray(dim1: dim1, dim2: dim2, dim3: dim3)
    
    let tensor = Tensor(data: values.flatMap { $0.flatMap { $0 } }, shape: [dim1, dim2, dim3])
    
    // Transpose dimensions 0 and 1
    var transposedTensor = tensor
    transposedTensor.transpose(0, 1)
    XCTAssertEqual(transposedTensor.transpose(dim0: 0, dim1: 1).nestedArray as! [[[Float]]], values)
    XCTAssertEqual(transposedTensor.transpose(dim0: 0, dim1: 2).data,  tensor.data)
  }
  
}
