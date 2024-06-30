//
//  TensorAddition.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation

//
//  stackTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/11/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorCPUAdditionTests: XCTestCase {
  func testSingleElementVectorAddition() {
    let tensors: [Tensor<Float>] = Array(repeating: rand([1]), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 + t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 + el.1 == output.data[idx]
    })
  }
  
  func test1DAddition() {
    let M: Int = Int.random(in: 1...100)
    let tensors: [Tensor<Float>] = Array(repeating: rand([M]), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 + t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 + el.1 == output.data[idx]
    })
  }
  
  func test2DAddition() {
    let M: Int = Int.random(in: 1...100)
    let P: Int = Int.random(in: 1...100)
    let tensors: [Tensor<Float>] = Array(repeating: rand([M, P]), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 + t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 + el.1 == output.data[idx]
    })
  }
  
  func testNDAddition() {
    let dims: [Int] = Array(repeating: Int.random(in: 1...10), count: 4)
    let tensors: [Tensor<Float>] = Array(repeating: rand(dims), count: 2)
    let t1: Tensor<Float> = tensors[0]
    let t2: Tensor<Float> = tensors[1]
    let output: Tensor<Float> = t1 + t2
    XCTAssert(zip(t1.data, t2.data).enumerated().allSatisfy {
      (idx, el) in el.0 + el.1 == output.data[idx]
    })
  }
  
  func testBroadcastAddition() {
    let t1: Tensor<Float> = Tensor(
      [[1.0, 2.0, 3.0, 4.0]]
    )
    let t2: Tensor<Float> = Tensor(
      [[-1, -2.0,-3.0,-4.0],
       [110.0,210.0,310.0,410.0],
       [0.0,0.0,0.0,0]
      ]
    )
    let t3: [[Float]] =
    [[0.0, 0.0, 0.0, 0.0],
     [111.0, 212.0, 313.0, 414],
     [1.0,2.0,3.0, 4.0]
    ]
    let output: Tensor<Float> = t1 + t2
    XCTAssertEqual(output.nestedArray as! [[Float]], t3)
  }
  
  func testBroadcastAddition2() {
    let t1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3,1])
    let t2: Tensor<Float> = Tensor(
      [[-1, -2.0,-3.0,-4.0],
       [110.0,210.0,310.0,410.0],
       [0.0,0.0,0.0,0]
      ]
    )
    let t3: [[Float]] =
    [[0.0, -1.0, -2.0, -3.0],
     [112.0, 212.0, 312.0, 412],
     [3.0, 3.0, 3.0, 3.0]
    ]
    let output: Tensor<Float> = t1 + t2
    XCTAssertEqual(output.nestedArray as! [[Float]], t3)
  }
  
  func testBroadcastAddition3() {
    let t1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3,1])
    let t2: Tensor<Float> = zeros(3, 3)
    let t3: [[Float]] =
    [
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3]
    ]
    let output: Tensor<Float> = t1 + t2
    XCTAssertEqual(output.nestedArray as! [[Float]], t3)
    
    let t4: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
    let t5: Tensor<Float> = zeros(3, 3)
    let t6: [[Float]] =
    [
      [1, 2, 3],
      [1, 2, 3],
      [1, 2, 3]
    ]
    let output2: Tensor<Float> = t4 + t5
    XCTAssertEqual(output2.nestedArray as! [[Float]], t6)
  }
}
