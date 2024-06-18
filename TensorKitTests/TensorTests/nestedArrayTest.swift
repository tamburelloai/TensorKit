//
//  nestedArrayTest.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//

import XCTest
@testable import TensorKit

extension TensorTests {
  /// `Placeholder for test. Already heavily tested though in TensorInitTests`
  func testNestedArray1D() {
    let nestedArr1D: [Float] = [1, 2, 3]
    let tensor1D: Tensor<Float> = Tensor(nestedArr1D)
    XCTAssertEqual(nestedArr1D, tensor1D.nestedArray() as! [Float])
  }
  func testNestedArray2D() {
    let nestedArr2D: [[Float]] = [[1, 2, 3],[4, 5, 6]]
    let tensor2D: Tensor<Float> = Tensor(nestedArr2D)
    XCTAssertEqual(nestedArr2D, tensor2D.nestedArray() as! [[Float]])
  }
  
  func testNestedArray3D() {
    let nestedArr3D: [[[Float]]] = [
      [[1, 2, 3],
       [4, 5, 6]],
      
      [[1, 2, 3],
       [4, 5, 6]],
    ]
    let tensor3D: Tensor<Float> = Tensor(nestedArr3D)
    XCTAssertEqual(nestedArr3D, tensor3D.nestedArray() as! [[[Float]]])
  }
}
