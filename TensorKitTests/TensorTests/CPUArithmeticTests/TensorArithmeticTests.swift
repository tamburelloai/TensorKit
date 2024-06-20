//
//  TensorArithmeticTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//


import XCTest
@testable import TensorKit

final class TensorArithmeticTests: XCTestCase {
  
  override func setUpWithError() throws {
    // Put setup code here. This method is called before the invocation of each test method in the class.
  }
  
  override func tearDownWithError() throws {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
  }
  
  func testExample() throws {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    // Any test you write for XCTest can be annotated as throws and async.
    // Mark your test throws to produce an unexpected failure when your test encounters an uncaught error.
    // Mark your test async to allow awaiting for asynchronous code to complete. Check the results with assertions afterwards.
  }
  
  func testPerformanceExample() throws {
    // This is an example of a performance test case.
    self.measure {
      // Put the code you want to measure the time of here.
    }
  }
  
  func testTensorIntAdditionOperator() {
    let tensor1: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor2: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor3: Tensor<Float> = tensor1 + tensor2
    XCTAssertEqual(tensor3.data, [2, 4, 6, 8])
  }
  
  func testTensorIntSubtractionOperator() {
    let tensor1: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor2: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor3: Tensor<Float> = tensor1 - tensor2
    XCTAssertEqual(tensor3.data, [0.0, 0.0, 0.0, 0.0])
  }
  
  func testTensorIntDivisionOperator() {
    let tensor1: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor2: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor3: Tensor<Float> = tensor1 / tensor2
    XCTAssertEqual(tensor3.data, [1.0, 1.0, 1.0, 1.0])
  }
  
  func testTensorIntMultiplicationOperator() {
    let tensor1: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor2: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let tensor3: Tensor<Float> = tensor1 * tensor2
    XCTAssertEqual(tensor3.data, [1.0, 4.0, 9.0, 16.0])
  }
  
  
  
}
