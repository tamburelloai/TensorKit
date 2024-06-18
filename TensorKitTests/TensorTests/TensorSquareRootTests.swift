//
//  TensorSquareRootTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/15/24.
//

import Foundation
import XCTest
@testable import TensorKit

class TensorSquareRootTests: XCTestCase {
  
  func testSquareRoot1D() {
    let tensor = Tensor<Float>([1.0, 2, 3, 4])
    let output = Tensor.sqrt(tensor)
    let trueValues: [Float] = [1.0, 1.41421356, 1.73205081, 2.0]
    for i in (0..<trueValues.count) {
      XCTAssertEqual(output.data[i], trueValues[i], accuracy: 0.99)
    }

  }
  
  func testSquareRoot2D() {
    let tensor = Tensor<Float>([[1.0, 2, 3, 4], [1, 9, 1, 1]])
    let output = Tensor.sqrt(tensor)
    let trueValues: [Float] = [1.0, 1.41421356, 1.73205081, 2.0, 1, 3, 1, 1]
    for i in (0..<trueValues.count) {
      XCTAssertEqual(output.data[i], trueValues[i], accuracy: 0.99)
    }

  }
  
}
