//
//  eyeTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//

import XCTest
@testable import TensorKit

final class eyeTests: XCTestCase {
  
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
  
  func testTensorEye() {
    // testing individual integer init
    let N: Int = Int.random(in: 1...10)
    let tensor: Tensor<Int> = eye(N)
    XCTAssertEqual(tensor.shape, [N, N])
    XCTAssertEqual(tensor.data.count, N*N)
    XCTAssertEqual(tensor.data.reduce(0, +), N)
    XCTAssertEqual(tensor.data[0], 1)

    let P: Int = Int.random(in: 1...10)
    let floatTensor: Tensor<Float> = eye(P)
    XCTAssertEqual(floatTensor.shape, [P, P])
    XCTAssertEqual(floatTensor.data.count, P*P)
    XCTAssertEqual(floatTensor.data.reduce(0, +), Float(P))
    XCTAssertEqual(floatTensor.data[0], Float(1.0))

    let Q: Int = 3
    let boolTensor : Tensor<Bool> = eye(Q)
    XCTAssertEqual(boolTensor.shape, [Q, Q])
    XCTAssertEqual(boolTensor.data.count, Q*Q)
    XCTAssertEqual(boolTensor.nestedArray() as! [[Bool]], [[true, false, false],
                                                            [false, true, false],
                                                            [false, false, true]])

  }
  
}
