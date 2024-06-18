//
//  TensorVarianceTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/14/24.
//

import XCTest
@testable import TensorKit

final class TensorVarianceTests: XCTestCase {
  
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
  
  func testGlobalVariance1DValue() {
    let tensor: Tensor<Float> = Tensor([1.0, 2, 3, 4])
    let tensorVariance: Tensor<Float> = tensor.variance()
    XCTAssertEqual(tensorVariance.item(), 1.6667, accuracy: 0.999)
  }
  
  func testGlobalVariance2DValue() {
    let tensor: Tensor<Float> = Tensor([[1.0, 2], [3, 4]])
    let tensorVariance: Tensor<Float> = tensor.variance()
    XCTAssertEqual(tensorVariance.item(), 1.6667, accuracy: 0.999)
  }
  
  func testDimVariance1DValue() {
    let tensor: Tensor<Float> = Tensor([1.0, 2, 3, 4])
    let tensorVariance: Tensor<Float> = tensor.variance(dim: 0)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [0])
    XCTAssertEqual(tensorVariance.item(), tensorVarianceSameDifferentSig.item())
    XCTAssertEqual(tensorVariance.item(), 1.6667, accuracy: 0.999)
  }
  
  func testDimVariance1DValue_KEEPDIM() {
    let tensor: Tensor<Float> = Tensor([1.0, 2, 3, 4])
    let tensorVariance: Tensor<Float> = tensor.variance(dim: 0, keepDim: true)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [0], keepDim: true)
    XCTAssertEqual(tensorVariance.item(), tensorVarianceSameDifferentSig.item())
    XCTAssertEqual(tensorVariance.item(), 1.6667, accuracy: 0.999)
    XCTAssertEqual(tensorVariance.shape, [1])
    XCTAssertEqual(tensorVarianceSameDifferentSig.shape, [1])
  }
  
  
  
  func test_Dim0_Variance_2D() {
    let dim: Int = 0
    let keepDimBool: Bool = false
    let trueShape: [Int] = [2]
    let trueValue: [Float] = [2.0, 2.0]
    let tensor: Tensor<Float> = Tensor(
      [[1.0, 2.0],
       [3.0, 4.0]]
    )
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    XCTAssertEqual(tensorVariance.data, trueValue)
    XCTAssertEqual(tensorVariance.shape, trueShape)
  }
  
  func test_Dim0_Variance_2D_KEEPDIM() {
    let dim: Int = 0
    let keepDimBool: Bool = true
    let trueShape: [Int] = [1, 2]
    let trueValue: [Float] = [2.0, 2.0]
    let tensor: Tensor<Float> = Tensor(
      [[1.0, 2.0],
       [3.0, 4.0]]
    )
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    XCTAssertEqual(tensorVariance.data, trueValue)
    XCTAssertEqual(tensorVariance.shape, trueShape)
  }
  
  func test_Dim1_Variance_2D() {
    let dim: Int = 1
    let keepDimBool: Bool = false
    let trueShape: [Int] = [2]
    let trueValue: [Float] = [0.5, 0.5]
    let tensor: Tensor<Float> = Tensor(
      [[1.0, 2.0],
       [3.0, 4.0]]
    )
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    XCTAssertEqual(tensorVariance.data, trueValue)
    XCTAssertEqual(tensorVariance.shape, trueShape)
  }
  
  func test_Dim1_Variance_2D_KEEPDIM() {
    let dim: Int = 1
    let keepDimBool: Bool = true
    let trueShape: [Int] = [2, 1]
    let trueValue: [Float] = [0.5, 0.5]
    let tensor: Tensor<Float> = Tensor(
      [[1.0, 2.0],
       [3.0, 4.0]]
    )
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    XCTAssertEqual(tensorVariance.data, trueValue)
    XCTAssertEqual(tensorVariance.shape, trueShape)
  }
  
  func test_Dim0_Variance_2D_AGAIN() {
    let dim: Int = 0
    let keepDimBool: Bool = false
    let trueOutputShape: [Int] = [4]
    let trueOutputValue: [Float] = [1.5926, 1.0056, 1.2005, 0.3646]
    let tensor: Tensor<Float> = Tensor(
      [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]]
    )
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    for i in(0..<tensorVariance.data.count) {
      XCTAssertEqual(tensorVariance.data[i], trueOutputValue[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensorVariance.shape, trueOutputShape)
  }
  
  func test_Dim0_Variance_2D_KEEPDIM_AGAIN() {
    let dim: Int = 0
    let keepDimBool: Bool = true
    let trueOutputShape: [Int] = [1, 4]
    let trueOutputValue: [Float] = [1.5926, 1.0056, 1.2005, 0.3646]
    let tensor: Tensor<Float> = Tensor(
      [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]]
    )
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    for i in(0..<tensorVariance.data.count) {
      XCTAssertEqual(tensorVariance.data[i], trueOutputValue[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensorVariance.shape, trueOutputShape)
  }
  
  func test_Dim1_Variance_2D_AGAIN() {
    let dim: Int = 1
    let keepDimBool: Bool = false
    let tensor: Tensor<Float> = Tensor(
      [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]]
    )
    let trueOutputShape: [Int] = [4]
    let trueOutputValue: [Float] = [1.0631, 0.5590, 1.4893, 0.8258]
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    for i in(0..<tensorVariance.data.count) {
      XCTAssertEqual(tensorVariance.data[i], trueOutputValue[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensorVariance.shape, trueOutputShape)
  }
  
  func test_Dim1_Variance_2D_KEEPDIM_AGAIN() {
    let dim: Int = 1
    let keepDimBool: Bool = true
    let tensor: Tensor<Float> = Tensor(
      [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]]
    )
    let trueOutputShape: [Int] = [4, 1]
    let trueOutputValue: [Float] = [1.0631, 0.5590, 1.4893, 0.8258]
    let tensorVariance: Tensor<Float> = tensor.variance(dim: dim, keepDim: keepDimBool)
    let tensorVarianceSameDifferentSig: Tensor<Float> = tensor.variance(dims: [dim], keepDim: keepDimBool)
    XCTAssertEqual(tensorVariance.data, tensorVarianceSameDifferentSig.data)
    XCTAssertEqual(tensorVariance.shape, tensorVarianceSameDifferentSig.shape)
    for i in(0..<tensorVariance.data.count) {
      XCTAssertEqual(tensorVariance.data[i], trueOutputValue[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensorVariance.shape, trueOutputShape)
  }
  
  
  
  
  
  
}

