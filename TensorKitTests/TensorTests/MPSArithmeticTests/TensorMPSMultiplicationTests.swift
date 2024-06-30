//
//  TensorMPSMultiplication.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/17/24.
//


import Foundation
import XCTest
@testable import TensorKit

final class TensorMPSMultiplicationTests: XCTestCase {
  func testMetalInit() {
    let backend = MPSBackend.shared
    XCTAssertNotNil(backend)
  }
  
  func testMetalLibrary() {
    let backend = MPSBackend.shared
    let library = backend.defaultLibrary
    XCTAssertNotNil(library)
  }
  
  func testSingleElementVectorMultiplication() {
    let tensors: [Tensor<Float>] = Array(repeating: rand([1]).to(.mps), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 * t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 * el.1 == output.data[idx]
    })
  }
  
  func test2DMultiplication() {
    let M: Int = Int.random(in: 1...100)
    let P: Int = Int.random(in: 1...100)
    let tensors: [Tensor<Float>] = Array(repeating: rand([M, P]).to(.mps), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 * t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 * el.1 == output.data[idx]
    })
    XCTAssertEqual(output.device, .mps)
  }
  
  func testNDMultiplication() {
    let dims: [Int] = [10, 4, 12]
    let tensors: [Tensor<Float>] = Array(repeating: rand(dims).to(.mps), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 * t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 * el.1 == output.data[idx]
    })
  }
  
    func testBroadcastMultiplication() {
      let a: Tensor<Float> = Tensor(
        [[1.0, 2.0, 3.0, 4.0]]
      ).to(.mps)
  
      let b: Tensor<Float> = Tensor(
        [[-1, -2.0,-3.0,-4.0],
         [110.0,210.0,310.0,410.0],
         [0.0,0.0,0.0,0.0]
        ]
      ).to(.mps)
      let ab: [[Float]] =
       [[-1, -4.0,-9.0,-16.0],
        [110.0, 420.0,930.0, 1640.0],
        [0.0,0.0,0.0,0.0]
      ]
      let output: Tensor<Float> = a * b
      XCTAssertEqual(output.nestedArray as! [[Float]], ab)
    }
  
    func testBroadcastMultiplication2() {
      var a: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3,1]).to(.mps)
      var b: Tensor<Float> = Tensor(
        [[-1, -2.0,-3.0,-4.0],
         [110.0,210.0,310.0,410.0],
         [0.0,0.0,0.0,0]
        ]).to(.mps)
      var ab: [[Float]] =
      [[-1.0, -2.0, -3.0, -4.0],
       [220.0, 420.0, 620.0, 820.0],
       [0.0, 0.0, 0.0, 0.0]
      ]
      var output: Tensor<Float> = a*b
      XCTAssertEqual(output.nestedArray as! [[Float]], ab)
    }
  
    func testBroadcastMultiplication3() {
      let t1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3,1]).to(.mps)
      let t2: Tensor<Float> = ones(3, 3).to(.mps)
      let t3: [[Float]] =
      [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
      ]
      let output: Tensor<Float> = t1 * t2
      XCTAssertEqual(output.nestedArray as! [[Float]], t3)
  
      let t4: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
      let t5: Tensor<Float> = ones(3, 3)
      let t6: [[Float]] =
      [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
      ]
      let output2: Tensor<Float> = t4 * t5
      XCTAssertEqual(output2.nestedArray as! [[Float]], t6)
    }
  
  
}
