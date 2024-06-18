//
//  TensorMPSSubtractionTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorMPSSubtractionTests: XCTestCase {
  func testMetalInit() {
    let backend = MPSBackend.shared
    XCTAssertNotNil(backend)
  }
  
  func testMetalLibrary() {
    let backend = MPSBackend.shared
    let library = backend.defaultLibrary
    XCTAssertNotNil(library)
  }
  
  func testSingleElementVectorSubtraction() {
    let tensors: [Tensor<Float>] = Array(repeating: rand([1]).to(.mps), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 - t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 - el.1 == output.data[idx]
    })
  }
  
  func test2DSubtraction() {
    let M: Int = Int.random(in: 1...100)
    let P: Int = Int.random(in: 1...100)
    let tensors: [Tensor<Float>] = Array(repeating: rand([M, P]).to(.mps), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 - t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 - el.1 == output.data[idx]
    })
  }
  
  func testNDSubtraction() {
    let dims: [Int] = [2, 4, 5, 1]
    let tensors: [Tensor<Float>] = Array(repeating: rand(dims).to(.mps), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 - t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 - el.1 == output.data[idx]
    })
  }
  
  
  func testBroadcastSubtraction3() {
    let t1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3,1]).to(.mps)
    let t2: Tensor<Float> = zeros(3, 3).to(.mps)
    let t3: [[Float]] =
    [
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3]
    ]
    let output: Tensor<Float> = t1 - t2
    XCTAssertEqual(output.nestedArray() as! [[Float]], t3)
    
    let t4: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
    let t5: Tensor<Float> = zeros(3, 3)
    let t6: [[Float]] =
    [
      [1, 2, 3],
      [1, 2, 3],
      [1, 2, 3]
    ]
    let output2: Tensor<Float> = t4 - t5
    XCTAssertEqual(output2.nestedArray() as! [[Float]], t6)
  }
  
  
}
