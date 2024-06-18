//
//  TensorBroadcastTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/10/24.
//

import Foundation
import XCTest
@testable import TensorKit


final class TensorBroadcastTests: XCTestCase {
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
  
  func testAllBroadcastCases() {
    let broadcastResults: [[String: Any]] = [
      [
        "array1_shape": [3],
        "array2_shape": [3, 1],
        "broadcasted_shape": [3, 3],
        "values": [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
        "flat_values": [2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
        "status": "success"
      ],
      [
        "array1_shape": [3],
        "array2_shape": [2],
        "status": "error",
        "error_message": "operands could not be broadcast together with shapes (3,) (2,)"
      ],
      [
        "array1_shape": [2, 2],
        "array2_shape": [2],
        "broadcasted_shape": [2, 2],
        "values": [[2, 4], [4, 6]],
        "flat_values": [2.0, 4.0, 4.0, 6.0],
        "status": "success"
      ],
      [
        "array1_shape": [1],
        "array2_shape": [3],
        "broadcasted_shape": [3],
        "values": [3, 4, 5],
        "flat_values": [3.0, 4.0, 5.0],
        "status": "success"
      ],
      [
        "array1_shape": [2, 2],
        "array2_shape": [2, 1],
        "broadcasted_shape": [2, 2],
        "values": [[6, 7], [9, 10]],
        "flat_values": [6.0, 7.0, 9.0, 10.0],
        "status": "success"
      ],
      [
        "array1_shape": [3],
        "array2_shape": [2, 2],
        "status": "error",
        "error_message": "operands could not be broadcast together with shapes (3,) (2,2)"
      ],
      [
        "array1_shape": [4],
        "array2_shape": [1, 4],
        "broadcasted_shape": [1, 4],
        "values": [[1.155480224167229, 0.9039647161015686, 0.8962199913660855, 0.8620468246140497]],
        "flat_values": [1.155480224167229, 0.9039647161015686, 0.8962199913660855, 0.8620468246140497],
        "status": "success"
      ],
      [
        "array1_shape": [1, 2, 2],
        "array2_shape": [2, 1],
        "broadcasted_shape": [1, 2, 2],
        "values": [[[0.9571414479483599, 0.8902305456871449], [1.811238187780702, 1.7567750931826145]]],
        "flat_values": [0.9571414479483599, 0.8902305456871449, 1.811238187780702, 1.7567750931826145],
        "status": "success"
      ],
      [
        "array1_shape": [2],
        "array2_shape": [4, 4],
        "status": "error",
        "error_message": "operands could not be broadcast together with shapes (2,) (4,4)"
      ],
      [
        "array1_shape": [4, 2],
        "array2_shape": [4, 3],
        "status": "error",
        "error_message": "operands could not be broadcast together with shapes (4,2) (4,3)"
      ],
      [
        "array1_shape": [4],
        "array2_shape": [3, 1],
        "broadcasted_shape": [3, 4],
        "values": [[1.1324704722665695, 1.1832840075556894, 1.4815693792195659, 1.387817848067077], [0.6785607892482758, 0.7293743245373957, 1.027659696201272, 0.9339081650487833], [1.0740053100675975, 1.1248188453567174, 1.4231042170205939, 1.329352685868105]],
        "flat_values": [1.1324704722665695, 1.1832840075556894, 1.4815693792195659, 1.387817848067077, 0.6785607892482758, 0.7293743245373957, 1.027659696201272, 0.9339081650487833, 1.0740053100675975, 1.1248188453567174, 1.4231042170205939, 1.329352685868105],
        "status": "success"
      ],
      [
        "array1_shape": [3],
        "array2_shape": [4, 3, 4],
        "status": "error",
        "error_message": "operands could not be broadcast together with shapes (3,) (4,3,4)"
      ],
      [
        "array1_shape": [1],
        "array2_shape": [3, 1],
        "broadcasted_shape": [3, 1],
        "values": [[0.433756390178481], [0.7054389749237151], [0.7263468254706235]],
        "flat_values": [0.433756390178481, 0.7054389749237151, 0.7263468254706235],
        "status": "success"
      ],
      [
        "array1_shape": [1, 1, 2],
        "array2_shape": [4],
        "status": "error",
        "error_message": "operands could not be broadcast together with shapes (1,1,2) (4,)"
      ],
      [
        "array1_shape": [4],
        "array2_shape": [1],
        "broadcasted_shape": [4],
        "values": [0.8767342342254862, 1.1613399412442413, 0.31194123933436857, 1.0306189920445892],
        "flat_values": [0.8767342342254862, 1.1613399412442413, 0.31194123933436857, 1.0306189920445892],
        "status": "success"
      ],
      [
        "array1_shape": [4, 2],
        "array2_shape": [1, 2, 2],
        "status": "error",
        "error_message": "operands could not be broadcast together with shapes (4,2) (1,2,2)"
      ]
    ]
    for res in broadcastResults {
      let shapeA: [Int] = res["array1_shape"] as! [Int]
      let dataA: [Float] = initZeros(shapeA)
      let tensorA: Tensor = Tensor(data: dataA, shape: shapeA)
      let shapeB: [Int] = res["array2_shape"] as! [Int]
      let dataB: [Float] = initZeros(shapeB)
      let tensorB: Tensor = Tensor(data: dataB, shape: shapeB)
      if res["status"] as! String == "success" {
        let tensorC: Tensor = tensorA + tensorB
        XCTAssertEqual(tensorC.shape, res["broadcasted_shape"] as! [Int])
        XCTAssertEqual(tensorC.data.count, (res["flat_values"] as! [Double]).count)
      } else if res["status"] as! String == "error"{
        XCTAssertEqual(false, broadcastCompatible(tensorA, tensorB))
      }
    }
  }
  
}
