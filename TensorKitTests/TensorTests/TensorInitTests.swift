//
//  TensorNDArrayInitTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//

import XCTest
@testable import TensorKit

final class TensorInitTests: XCTestCase {
  
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
  
  func test_Bool_1D_ArrayInit() {
    let values: [Bool] = [false, false, false, false]
    let boolTensor: Tensor<Bool> = Tensor(values)
    XCTAssertEqual(boolTensor.shape, [4])
    XCTAssertEqual(boolTensor.strides, [1])
    XCTAssertEqual(boolTensor.data, [false, false, false, false])
  }
  
  func test_Bool_2D_ArrayInit() {
    let values: [[Bool]] = [
      [false, false, false, false],
      [false, false, false, false]
    ]
    let boolTensor: Tensor<Bool> = Tensor(values)
    XCTAssertEqual(boolTensor.shape, [2, 4])
    XCTAssertEqual(boolTensor.strides, [4, 1])
    XCTAssertEqual(boolTensor.data, [false, false, false, false, false, false, false, false])
  }
  
  func test_Bool_3D_ArrayInit() {
    let values: [[[Bool]]] = [
      [
        [false, false, false, false],
        [false, false, false, false]
      ],
      [
        [false, false, false, false],
        [false, false, false, false]
      ],
      [
        [false, false, false, false],
        [false, false, false, false]
      ]
    ]
    
    let boolTensor: Tensor<Bool> = Tensor(values)
    XCTAssertEqual(boolTensor.shape, [3, 2, 4])
    XCTAssertEqual(boolTensor.strides, [8, 4, 1])
    XCTAssertEqual(boolTensor.data, [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false])
  }
  
  func test_Int_1D_ArrayInit() {
    let intTensor: Tensor<Int> = Tensor([1, 2, 3, 4])
    XCTAssertEqual(intTensor.shape, [4])
    XCTAssertEqual(intTensor.strides, [1])
    XCTAssertEqual(intTensor.data, [1, 2, 3, 4])
  }
  
  func test_Int_2D_ArrayInit() {
    let intTensor: Tensor<Int> = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    XCTAssertEqual(intTensor.shape, [2, 4])
    XCTAssertEqual(intTensor.strides, [4, 1])
    XCTAssertEqual(intTensor.data, [1, 2, 3, 4, 5, 6, 7, 8])
  }
  
  func test_Int_3D_ArrayInit() {
    let intTensor: Tensor<Int> = Tensor(
      [
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        
        [[9, 10, 11, 12],
         [13, 14, 15, 16]],
        
        [[17, 18, 19, 20],
         [21, 22, 23, 24]]
      ])
    XCTAssertEqual(intTensor.shape, [3, 2, 4])
    XCTAssertEqual(intTensor.strides, [8, 4, 1])
    XCTAssertEqual(intTensor.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
  }
  
  func test_Double_1D_ArrayInit() {
    let values: [Double] = [1.0, 2.0, 3.0, 4.0]
    let floatTensor: Tensor<Float> = Tensor(values)
    XCTAssertEqual(floatTensor.shape, [4])
    XCTAssertEqual(floatTensor.strides, [1])
    XCTAssertEqual(floatTensor.data, [1, 2, 3, 4])
  }
  
  func test_Double_2D_ArrayInit() {
    let values: [[Double]] = [[1, 2, 3, 4], [5, 6, 7, 8]]
    let floatTensor: Tensor<Float> = Tensor(values)
    XCTAssertEqual(floatTensor.shape, [2, 4])
    XCTAssertEqual(floatTensor.strides, [4, 1])
    XCTAssertEqual(floatTensor.data, [1, 2, 3, 4, 5, 6, 7, 8])
  }
  
  func test_Double_3D_ArrayInit() {
    let values: [[[Double]]] = [
      [[1, 2, 3, 4],
       [5, 6, 7, 8]],
      
      [[9, 10, 11, 12],
       [13, 14, 15, 16]],
      
      [[17, 18, 19, 20],
       [21, 22, 23, 24]]
    ]
    let floatTensor: Tensor<Float> = Tensor(values)
    XCTAssertEqual(floatTensor.shape, [3, 2, 4])
    XCTAssertEqual(floatTensor.strides, [8, 4, 1])
    XCTAssertEqual(floatTensor.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
  }
  
  func test_Float_1D_ArrayInit() {
    let values: [Float] = [1.0, 2.0, 3.0, 4.0]
    let floatTensor: Tensor<Float> = Tensor(values)
    XCTAssertEqual(floatTensor.shape, [4])
    XCTAssertEqual(floatTensor.strides, [1])
    XCTAssertEqual(floatTensor.data, [1, 2, 3, 4])
  }
  
  func test_Float_2D_ArrayInit() {
    let values: [[Float]] = [[1, 2, 3, 4], [5, 6, 7, 8]]
    let floatTensor: Tensor<Float> = Tensor(values)
    XCTAssertEqual(floatTensor.shape, [2, 4])
    XCTAssertEqual(floatTensor.strides, [4, 1])
    XCTAssertEqual(floatTensor.data, [1, 2, 3, 4, 5, 6, 7, 8])
  }
  
  func test_Float_3D_ArrayInit() {
    let values: [[[Float]]] = [
      [[1, 2, 3, 4],
       [5, 6, 7, 8]],
      
      [[9, 10, 11, 12],
       [13, 14, 15, 16]],
      
      [[17, 18, 19, 20],
       [21, 22, 23, 24]]
    ]
    let floatTensor: Tensor<Float> = Tensor(values)
    XCTAssertEqual(floatTensor.shape, [3, 2, 4])
    XCTAssertEqual(floatTensor.strides, [8, 4, 1])
    XCTAssertEqual(floatTensor.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
  }
  
  
  func testTensorBoolInitialization() {
    let testData: [Bool] = [false, false, false]
    let tensor: Tensor<Bool> = Tensor(data: testData, shape: [3])
    XCTAssertEqual(tensor.data, testData)
  }
  
  
  func testTensorIntInitialization() {
    let testData: [Int] = [1, 2, 3]
    let tensor: Tensor<Int> = Tensor(data: testData, shape: [3])
    XCTAssertEqual(tensor.data, testData)
  }
  
  
  func testTensorFloatInitialization() {
    let testData: [Float] = [1.0, 0.5, 0.25]
    let tensor: Tensor<Float> = Tensor(data: testData, shape: [3])
    XCTAssertEqual(tensor.data, testData)
  }
  
  func testTensorDoubleInitialization() {
    let testData: [Double] = [1.0, 0.5, 0.25]
    let tensor: Tensor<Float> = Tensor(data: testData, shape: [3])
    for i in 0..<testData.count {
      XCTAssertEqual(tensor.data[i], Float(testData[i]))
    }
  }
}
