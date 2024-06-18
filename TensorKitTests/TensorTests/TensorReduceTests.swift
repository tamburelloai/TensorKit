//
//  TensorReduceTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/13/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorReduceTests: XCTestCase {
  func testReduce2DTensor() {
    let tensor: Tensor<Float> = Tensor([
      [1.0, 2, 3],
      [1, 2, 3],
      [1, 2, 4.0]])
    let resultTensor1: Tensor<Float> = tensor.reduce(+,  dim: 0)
    XCTAssertEqual(resultTensor1.data, [3.0, 6.0, 10.0])
    let resultTensor2: Tensor<Float> = tensor.reduce(+,  dim: 1)
    XCTAssertEqual(resultTensor2.data, [6.0, 6.0, 7.0])
  }

  func testReduce3DTensor() {
      // Define a 3D tensor
      let tensor: Tensor<Float> = Tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]],
        
        [[10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0],
         [16.0, 17.0, 18.0]],
        
        [[19.0, 20.0, 21.0],
         [22.0, 23.0, 24.0],
         [25.0, 26.0, 27.0]]])
      
      // Reduce along dimension 0
      let resultTensor0: Tensor<Float> = tensor.reduce(+, dim: 0)
      XCTAssertEqual(resultTensor0.data, [
        [30.0, 33.0, 36.0],
        [39.0, 42.0, 45.0],
        [48.0, 51.0, 54.0]
      ].flatMap { $0 }) // Flatten the array for comparison
      
      // Reduce along dimension 1
      let resultTensor1: Tensor<Float> = tensor.reduce(+, dim: 1)
      XCTAssertEqual(resultTensor1.data, [
        [12.0, 15.0, 18.0],
        [39.0, 42.0, 45.0],
        [66.0, 69.0, 72.0]
      ].flatMap { $0 }) // Flatten the array for comparison
      
      // Reduce along dimension 2
      let resultTensor2: Tensor<Float> = tensor.reduce(+, dim: 2)
      XCTAssertEqual(resultTensor2.data, [
        [6.0],
        [15.0],
        [24.0],
        [33.0],
        [42.0],
        [51.0],
        [60.0],
        [69.0],
        [78.0]
      ].flatMap { $0 }) // Flatten the array for comparison
  }
}
