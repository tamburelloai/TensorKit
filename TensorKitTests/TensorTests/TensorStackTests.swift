//
//  stackTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/11/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorStackTests: XCTestCase {
    func testValidInputToStackBool() {
    let tensorsValid: [Tensor<Float>] = [
      rand(5, 10),
      rand(5, 10)
    ]
    let tensorsInvalid: [Tensor<Float>] = [
      rand(5, 10),
      rand(5, 8)
    ]
    XCTAssertTrue(validInputToStack(tensorsValid))
    if validInputToStack(tensorsInvalid) {
      XCTFail()
    }
  }
  
  func testGetStackShape() {
    let tensors: [Tensor<Float>] = Array(repeating: zeros(2, 5), count: 10)
    XCTAssertEqual(Tensor._getStackShape(tensors, dim: 0), [10, 2, 5])
  }
  

  func testStackOneDim() {
    let tensor1: Tensor<Float> = Tensor([1.0, 1.0, 1.0, 1.0])
    let tensor2: Tensor<Float> = Tensor([2.0, 2.0, 2.0, 2.0])
    let tensors: [Tensor<Float>] = [tensor1, tensor2]
    let stackedTensor: Tensor<Float> = Tensor.stack(tensors, dim: 0)
    XCTAssertEqual(stackedTensor.shape, [2, 4])
    XCTAssertEqual(stackedTensor.data, [1, 1, 1, 1, 2, 2, 2, 2])
  }
  
  func testStackTwoDim() {
    let tensor1: Tensor<Float> = Tensor([[1.0, 1.0], [1.0, 1.0]])
    let tensor2: Tensor<Float> = Tensor([[2.0, 2.0], [2.0, 2.0]])
    let tensor3: Tensor<Float> = Tensor([[3.0, 3.0], [3.0, 3.0]])
    let tensors: [Tensor<Float>] = [tensor1, tensor2, tensor3]
    let stackedTensor: Tensor<Float> = Tensor.stack(tensors, dim: 0)
    XCTAssertEqual(stackedTensor.shape, [3, 2, 2])
    XCTAssertEqual(stackedTensor.data, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    
  }
}
