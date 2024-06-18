//
//  TensorMeanTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/14/24.
//

import XCTest
@testable import TensorKit

final class TensorMeanTests: XCTestCase {

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
  
  func testGlobalMean1DValue() {
    let tensor: Tensor<Float> = Tensor([1.0, 2, 3, 4])
    XCTAssertEqual(tensor.mean().item(), 2.5)
  }
  
  func testGlobalMean2DValue() {
    let tensor: Tensor<Float> = Tensor([[1, 1, 1, 1], [1.0, 2, 3, 4]])
    XCTAssertEqual(tensor.mean().item(), 1.75)
  }
  
  func testGlobalMean3DValue() {
    let tensor: Tensor<Float> = Tensor(
      [
        [[1, 1, 1, 1],
         [1.0, 2, 3, 4]],
        
        [[1, 2, 4, 4],
        [0, 0, 0, 0]]
      ])
    XCTAssertEqual(tensor.mean().item(), 1.5625)
  }
  
  func testGlobalMean1DShape() {
    let N: Int = Int.random(in: 1...10)
    let shape: [Int] = [N]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensor: Tensor<Float> = tensor.mean()
    XCTAssertEqual(tensor.shape, [N])
    XCTAssertEqual(meanTensor.shape, [])
  }
  
  func testGlobalMean2DShape() {
    let M: Int = Int.random(in: 1...10)
    let N: Int = Int.random(in: 1...10)
    let shape: [Int] = [M, N]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensor: Tensor<Float> = tensor.mean()
    XCTAssertEqual(tensor.shape, [M, N])
    XCTAssertEqual(meanTensor.shape, [])
  }
  
  func testGlobalMean3DShape() {
    let M: Int = Int.random(in: 1...10)
    let N: Int = Int.random(in: 1...10)
    let P: Int = Int.random(in: 1...10)
    let shape: [Int] = [M, N, P]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensor: Tensor<Float> = tensor.mean()
    XCTAssertEqual(tensor.shape, [M, N, P])
    XCTAssertEqual(meanTensor.shape, [])
  }
 
  func testDimMean1DShape() {
    let N: Int = Int.random(in: 1...10)
    let shape: [Int] = [N]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensor: Tensor<Float> = tensor.mean(dim: 0)
    XCTAssertEqual(tensor.shape, [N])
    XCTAssertEqual(meanTensor.shape, [])
  }
  
  func testDimMean1DShape_KEEPDIM() {
    let N: Int = Int.random(in: 1...10)
    let shape: [Int] = [N]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensor: Tensor<Float> = tensor.mean(dim: 0, keepDim: true)
    XCTAssertEqual(tensor.shape, [N])
    XCTAssertEqual(meanTensor.shape, [1])
  }
  
  func testDimMean2DShape() {
    let M: Int = Int.random(in: 1...10)
    let N: Int = Int.random(in: 1...10)
    let shape: [Int] = [M, N]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensorDim0: Tensor<Float> = tensor.mean(dim: 0, keepDim: false)
    let meanTensorDim1: Tensor<Float> = tensor.mean(dim: 1, keepDim: false)
    XCTAssertEqual(tensor.shape, [M, N])
    XCTAssertEqual(meanTensorDim0.shape, [N])
    XCTAssertEqual(meanTensorDim1.shape, [M])
  }
  
  func testDimMean2DShape_KEEPDIM() {
    let M: Int = Int.random(in: 1...10)
    let N: Int = Int.random(in: 1...10)
    let shape: [Int] = [M, N]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensorDim0: Tensor<Float> = tensor.mean(dim: 0, keepDim: true)
    let meanTensorDim1: Tensor<Float> = tensor.mean(dim: 1, keepDim: true)
    XCTAssertEqual(tensor.shape, [M, N])
    XCTAssertEqual(meanTensorDim0.shape, [1, N])
    XCTAssertEqual(meanTensorDim1.shape, [M, 1])
  }
  
  func testDimMean3DShape() {
    let M: Int = Int.random(in: 1...10)
    let N: Int = Int.random(in: 1...10)
    let P: Int = Int.random(in: 1...10)
    let shape: [Int] = [M, N, P]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensorDim0: Tensor<Float> = tensor.mean(dim: 0, keepDim: false)
    let meanTensorDim1: Tensor<Float> = tensor.mean(dim: 1, keepDim: false)
    let meanTensorDim2: Tensor<Float> = tensor.mean(dim: 2, keepDim: false)
    XCTAssertEqual(tensor.shape, [M, N, P])
    XCTAssertEqual(meanTensorDim0.shape, [N, P])
    XCTAssertEqual(meanTensorDim1.shape, [M, P])
    XCTAssertEqual(meanTensorDim2.shape, [M, N])
  }
  
  func testDimMean3DShape_KEEPDIM() {
    let M: Int = Int.random(in: 1...10)
    let N: Int = Int.random(in: 1...10)
    let P: Int = Int.random(in: 1...10)
    let shape: [Int] = [M, N, P]
    let tensor: Tensor<Float> = rand(shape)
    let meanTensorDim0: Tensor<Float> = tensor.mean(dim: 0, keepDim: true)
    let meanTensorDim1: Tensor<Float> = tensor.mean(dim: 1, keepDim: true)
    let meanTensorDim2: Tensor<Float> = tensor.mean(dim: 2, keepDim: true)
    XCTAssertEqual(tensor.shape, [M, N, P])
    XCTAssertEqual(meanTensorDim0.shape, [1, N, P])
    XCTAssertEqual(meanTensorDim1.shape, [M,1, P])
    XCTAssertEqual(meanTensorDim2.shape, [M, N, 1])
  }
}

