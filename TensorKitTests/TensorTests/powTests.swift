//
//  powTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/21/24.
//

import Foundation
import XCTest
@testable import TensorKit

class powTests: XCTestCase {
  func testIntTensorWithIntExponent() {
    let tensor = Tensor<Int>(data: [1, 2, 3], shape: [3])
    let result = pow(tensor, 2)
    XCTAssertEqual(result.data, [1, 4, 9])
    XCTAssertEqual(result.shape, [3])
  }
  
  func testFloatTensorWithFloatExponent() {
    let tensor = Tensor<Float>(data: [1.0, 2.0, 3.0], shape: [3])
    let result = pow(tensor, 2.0)
    XCTAssertEqual(result.data, [1.0, 4.0, 9.0])
    XCTAssertEqual(result.shape, [3])
  }
  
  func testFloatTensorWithIntExponent() {
    let tensor = Tensor<Float>(data: [1.0, 2.0, 3.0], shape: [3])
    let result = pow(tensor, 2)
    XCTAssertEqual(result.data, [1.0, 4.0, 9.0])
    XCTAssertEqual(result.shape, [3])
  }
  
  func testIntTensorWithFloatExponent() {
    let tensor = Tensor<Int>(data: [1, 2, 3], shape: [3])
    let result = pow(tensor, 2.0)
    XCTAssertEqual(result.data, [1.0, 4.0, 9.0])
    XCTAssertEqual(result.shape, [3])
  }
  
  func testEdgeCasesInt() {
    // Test zero and negative exponents
    let tensor = Tensor<Int>(data: [1, 2, 0], shape: [3])
    let resultZeroExp = pow(tensor, 0)
    let resultNegativeExp = pow(tensor, -1)
    XCTAssertEqual(resultZeroExp.data, [1, 1, 1])
    XCTAssertEqual(resultNegativeExp.data, [1, 0, 0]) // Assuming the behavior for pow(0, -1) returns 0
    
    // Test for empty tensors
    let emptyTensor: Tensor<Float> = Tensor([])
    let resultEmpty = pow(emptyTensor, 3)
    XCTAssertTrue(resultEmpty.data.isEmpty)
    XCTAssertEqual(resultEmpty.shape, [])
  }
  
  func testEdgeCasesFloat() {
    // Test zero and negative exponents
    let tensor: Tensor<Float> = Tensor(data: [1.0, 2, 0], shape: [3])
    let resultZeroExp = pow(tensor, 0)
    let resultNegativeExp = pow(tensor, -1)
    XCTAssertEqual(resultZeroExp.data, [1, 1, 1])
    XCTAssertEqual(resultNegativeExp.data, [1.0, 0.5, Float.infinity]) // Assuming the behavior for pow(0, -1) returns 0
    
    // Test for empty tensors
    let emptyTensor: Tensor<Float> = Tensor([])
    let resultEmpty = pow(emptyTensor, 3)
    XCTAssertTrue(resultEmpty.data.isEmpty)
    XCTAssertEqual(resultEmpty.shape, [])
  }
}
