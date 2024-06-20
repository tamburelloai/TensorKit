//
//  processValuesTest.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//

import XCTest
@testable import TensorKit


class TensorProcessValuesTests: XCTestCase {
  
  func testProcessValuesWithIntegers() {
    let values = [[1, 2, 3], [4, 5, 6]]
    let result = Tensor<Float>.processValues(values: values)
    XCTAssertEqual(result.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    XCTAssertEqual(result.1, [2, 3])
  }
  
  func testProcessValuesWithDoubles() {
    let values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let result = Tensor<Float>.processValues(values: values)
    XCTAssertEqual(result.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    XCTAssertEqual(result.1, [2, 3])
  }
  
  func testProcessValuesWithMixedTypes() {
    let values: [Any] = [[1, 2.0, 3], [4.0, 5, 6.0]]
    let result = Tensor<Float>.processValues(values: values)
    XCTAssertEqual(result.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    XCTAssertEqual(result.1, [2, 3])
  }
  
  func testProcessValuesWithNestedArrays() {
    let values: [Any] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    let result = Tensor<Float>.processValues(values: values)
    XCTAssertEqual(result.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    XCTAssertEqual(result.1, [2, 2, 2])
  }
}

