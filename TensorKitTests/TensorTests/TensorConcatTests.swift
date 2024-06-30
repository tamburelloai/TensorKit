//
//  TensorConcatTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/11/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorConcatTests: XCTestCase {
  func testInputToConcatBool() {
    let tensors1: [Tensor<Float>] = [
      rand(5, 10),
      rand(5, 15),
      rand(5, 5)
    ]
    XCTAssertEqual(false, validInputToConcat(tensors1, dim: 0))
    XCTAssertEqual(true, validInputToConcat(tensors1, dim: 1))
    
    let tensors2: [Tensor<Float>] = [
      rand(5, 10),
      rand(4, 15),
      rand(5, 5)
    ]
    XCTAssertEqual(false, validInputToConcat(tensors2, dim: 0))
    XCTAssertEqual(false, validInputToConcat(tensors2, dim: 1))
    
    let tensors3: [Tensor<Float>] = [
      rand(5, 5),
      rand(5, 5),
      rand(5, 5)
    ]
    XCTAssertEqual(true, validInputToConcat(tensors3, dim: 1))
    XCTAssertEqual(true, validInputToConcat(tensors3, dim: 0))
    
    let tensors4: [Tensor<Float>] = [
      rand(10, 5, 5),
      rand(10, 5, 50),
      rand(10, 5, 12)
    ]
    XCTAssertEqual(false, validInputToConcat(tensors4, dim: 0))
    XCTAssertEqual(false, validInputToConcat(tensors4, dim: 1))
    XCTAssertEqual(true, validInputToConcat(tensors4, dim: 2))
  }
  
  
  func testGetConcatShape() {
    let tensors1: [Tensor<Float>] = [
      rand(5, 10),
      rand(5, 15),
      rand(5, 5)
    ]
    let dimToTest1: Int = 1
    XCTAssertEqual(Tensor._getConcatShape(tensors1, dim: dimToTest1)[dimToTest1], 30)
    
    let tensors2: [Tensor<Float>] = [
      rand(5, 5),
      rand(5, 5),
      rand(5, 5)
    ]
    XCTAssertEqual(Tensor._getConcatShape(tensors2, dim: 0)[0], 15)
    XCTAssertEqual(Tensor._getConcatShape(tensors2, dim: 1)[1], 15)

    let tensors3: [Tensor<Float>] = [
      rand(10, 5, 5),
      rand(10, 5, 50),
      rand(10, 5, 12)
    ]
    XCTAssertEqual(Tensor._getConcatShape(tensors3, dim: 2)[2], 67)

  }
  
  func testConcatOneDim() {
    let M: Int = Int.random(in: 0...100)
    let N: Int = Int.random(in: 0...100)
    let tensor1: Tensor<Float> = rand(M)
    let tensor2: Tensor<Float> = rand(N)
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 0)
    XCTAssertEqual(tensor3.shape, [M+N])
    XCTAssertEqual(tensor3.data.count, M+N)
  }
  
  func testConcatTwoDim() {
    let M: Int = Int.random(in: 0...100)
    let N: Int = Int.random(in: 0...100)
    let P: Int = Int.random(in: 0...100)
    let tensor1: Tensor<Float> = rand(M, N)
    let tensor2: Tensor<Float> = rand(M, P)
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 1)
    XCTAssertEqual(tensor3.shape, [M, N+P])
    XCTAssertEqual(tensor3.data.count, M*(N+P))
  }
  
  func testConcatOneDimValuePlacement() {
    let tensor1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0], shape: [3])
    let tensor2: Tensor<Float> = Tensor(data: [4.0, 5.0], shape: [2])
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 0)
    XCTAssertEqual(tensor3.shape[0], 5)
    XCTAssertEqual(tensor3[2], Float(3))
    XCTAssertEqual(tensor3[4], Float(5))
  }
  
  func testConcatTwoDimValuePlacement() {
    let tensor1: Tensor<Float> = Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3])
    let tensor2: Tensor<Float> = Tensor(data: [7.0, 8.0, 9.0], shape: [1, 3])
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 0)
    XCTAssertEqual(tensor3.shape, [3, 3])
    XCTAssertEqual(tensor3.data[0], Float(1.0))
    XCTAssertEqual(tensor3.data[6], Float(7.0))
    XCTAssertEqual(tensor3.data[7], Float(8.0))
    XCTAssertEqual(tensor3[0, 0], Float(1.0))
    XCTAssertEqual(tensor3[2, 0], Float(7.0))
    XCTAssertEqual(tensor3[2, 1], Float(8.0))
  }
  
  func testConcatTwoDimValuePlacementNestedInit() {
    let tensor1: Tensor<Float> = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let tensor2: Tensor<Float> = Tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 0)
    XCTAssertEqual(tensor3.shape, [5, 3])
    XCTAssertEqual(tensor3.data[0], Float(1.0))
    XCTAssertEqual(tensor3.data[6], Float(7.0))
    XCTAssertEqual(tensor3.data[7], Float(8.0))
    XCTAssertEqual(tensor3[0, 0], Float(1.0))
    XCTAssertEqual(tensor3[2, 0], Float(7.0))
    XCTAssertEqual(tensor3[2, 1], Float(8.0))
    XCTAssertEqual(tensor3[4, 2], Float(15.0))
  }
  
  func testConcatThreeDimValuePlacementNestedInit_dim2() {
    let tensor1: Tensor<Float> = Tensor(
      [
        [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]],
        
        [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]]
      ]
    )
    let tensor2: Tensor<Float> = Tensor(
      [
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0]],
        
        [[9.0, 10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0, 16.0]]
      ]
    )
    let correctOutputTensor: [[[Float]]] = [
      [[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
      [4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0]],
      
      [[1.0, 2.0, 3.0, 9.0, 10.0, 11.0, 12.0],
      [4.0, 5.0, 6.0, 13.0, 14.0, 15.0, 16.0]]
    ]
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 2)
    XCTAssertEqual(tensor3.shape, [2, 2, 7])
    XCTAssertEqual(tensor3.nestedArray as! [[[Float]]], correctOutputTensor)
  }
  
  func testConcatThreeDimValuePlacementNestedInit_dim1() {
    let tensor1: Tensor<Float> = Tensor(
      [
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ]
      ]
    )
    let tensor2: Tensor<Float> = Tensor(
      [
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 2.0, 3.0, 1.0, 500.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [7.0, 2.0, 2.0, 4.0, 5.0],
          [6.0, 7.0, 1.0, 9.0, 10.0],
          [6.0, 7.0, 3.0, 9.0, 10.0],
          [1.0, 2.0, 3.0, 5.0, 5.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 22.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [64.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 2.0, 335.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.02, 9.0, 100.0],
          [1.0, 2.0, 3.0, 4.0, 5.10]
        ],
      ]
    )
    
    let correctOutputTensor: [[[Float]]] =
      [
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 2.0, 3.0, 1.0, 500.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [7.0, 2.0, 2.0, 4.0, 5.0],
          [6.0, 7.0, 1.0, 9.0, 10.0],
          [6.0, 7.0, 3.0, 9.0, 10.0],
          [1.0, 2.0, 3.0, 5.0, 5.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 22.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [64.0, 7.0, 8.0, 9.0, 10.0],
          [1.0, 2.0, 335.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.02, 9.0, 100.0],
          [1.0, 2.0, 3.0, 4.0, 5.10]
        ],
      ]
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 1)
    XCTAssertEqual(tensor3.shape, [2, 13, 5])
    XCTAssertEqual(tensor3.nestedArray as! [[[Float]]], correctOutputTensor)
  }
  
  
  func testConcatThreeDimValuePlacementNestedInit_dim0() {
    let tensor1: Tensor<Float> = Tensor(
      [
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ]
      ]
    )
    
    let tensor2: Tensor<Float> = Tensor(
      [
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ]
      ]
    )
    
    let correctOutputTensor: [[[Float]]] =
      [
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [6.0, 7.0, 8.0, 9.0, 10.0],
          [6.0, 7.0, 8.0, 9.0, 10.0]
        ]
      ]
    
    let tensor3 = Tensor.concat([tensor1, tensor2], dim: 0)
    XCTAssertEqual(tensor3.shape, [12, 3, 5])
    XCTAssertEqual(tensor3.nestedArray as! [[[Float]]], correctOutputTensor)
  }
  
  
  
}
