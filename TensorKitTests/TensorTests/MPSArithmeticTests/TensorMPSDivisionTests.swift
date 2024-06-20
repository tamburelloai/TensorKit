//
//  TensorMPSDivisionTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/17/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorMPSDivisionTests: XCTestCase {
  func testMetalInit() {
    let backend = MPSBackend.shared
    XCTAssertNotNil(backend)
  }
  
  func testMetalLibrary() {
    let backend = MPSBackend.shared
    let library = backend.defaultLibrary
    XCTAssertNotNil(library)
  }
  
  func testSingleElementVectorDivision() {
    let tensors: [Tensor<Float>] = Array(repeating: rand([1]).to(.mps), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 / t2
    for i in (0..<output.data.count) {
      XCTAssertEqual(t1.data[i]/t2.data[i], output.data[i], accuracy: 0.999)
    }
  }
  
  func test2DDivision() {
    let dims: [Int] = [4, 12]
    let t1: Tensor<Float> = rand(dims).to(.mps)
    let t2: Tensor<Float> = rand(dims).to(.mps)
    let output: Tensor<Float> = t1 / t2
    for i in (0..<output.data.count) {
      XCTAssertEqual(t1.data[i]/t2.data[i], output.data[i], accuracy: 0.999)
    }
    XCTAssertEqual(output.device, .mps)
    XCTAssertEqual(output.shape, dims)
  }
  
  func testNDDivision() {
    let dims: [Int] = [10, 4, 12]
    let t1: Tensor<Float> = rand(dims).to(.mps)
    let t2: Tensor<Float> = rand(dims).to(.mps)
    let output: Tensor<Float> = t1 / t2
    for i in (0..<output.data.count) {
      XCTAssertEqual(t1.data[i]/t2.data[i], output.data[i], accuracy: 0.999)
    }
    XCTAssertEqual(output.device, .mps)
    XCTAssertEqual(output.shape, dims)
  }
  
//  func testBroadcastDivision() {
//    let t1: Tensor<Float> = Tensor(
//      [[1.0, 2.0, 3.0, 4.0]]
//    ).to(.mps)
//    
//    let t2: Tensor<Float> = Tensor(
//      [[-1, -2.0,-3.0,-4.0],
//       [110.0,210.0,310.0,410.0],
//       [0.0,0.0,0.0,0.0]
//      ]
//    ).to(.mps)
//    let t3: [[Float]] =
//    [[0.0, 0.0, 0.0, 0.0],
//     [111.0, 212.0, 313.0, 414],
//     [1.0,2.0,3.0, 4.0]
//    ]
//    let output: Tensor<Float> = t1 / t2
//    XCTAssertEqual(output.nestedArray() as! [[Float]], t3)
//  }
  
//  func testBroadcastDivision2() {
//    let t1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3,1]).to(.mps)
//    let t2: Tensor<Float> = Tensor(
//      [[-1, -2.0,-3.0,-4.0],
//       [110.0,210.0,310.0,410.0],
//       [0.0,0.0,0.0,0]
//      ]
//    ).to(.mps)
//    let t3: [[Float]] =
//    [[0.0, -1.0, -2.0, -3.0],
//     [112.0, 212.0, 312.0, 412],
//     [3.0, 3.0, 3.0, 3.0]
//    ]
//    let output: Tensor<Float> = t1 / t2
//    XCTAssertEqual(output.nestedArray() as! [[Float]], t3)
//  }
  
//  func testBroadcastDivision3() {
//    let t1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3,1]).to(.mps)
//    let t2: Tensor<Float> = zeros(3, 3).to(.mps)
//    let t3: [[Float]] =
//    [
//      [1, 1, 1],
//      [2, 2, 2],
//      [3, 3, 3]
//    ]
//    let output: Tensor<Float> = t1 / t2
//    XCTAssertEqual(output.nestedArray() as! [[Float]], t3)
//    
//    let t4: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
//    let t5: Tensor<Float> = zeros(3, 3)
//    let t6: [[Float]] =
//    [
//      [1, 2, 3],
//      [1, 2, 3],
//      [1, 2, 3]
//    ]
//    let output2: Tensor<Float> = t4 / t5
//    XCTAssertEqual(output2.nestedArray() as! [[Float]], t6)
//  }
//  
//  
}
