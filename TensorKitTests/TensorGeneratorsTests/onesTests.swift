//
//  onesTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//

import XCTest
@testable import TensorKit

final class onesTests: XCTestCase {
  
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
  
  func testTensorOnes() {
    var M: Int = Int.random(in: 1...10)
    var N: Int = Int.random(in: 1...10)
    let tensorInt: Tensor<Int> = ones(M, N)
    XCTAssertEqual(tensorInt.shape, [M, N])
    XCTAssertEqual(tensorInt.data.count, M*N)
    XCTAssertEqual(tensorInt.data.reduce(0, +), M*N)
    
    M = Int.random(in: 1...10)
    N = Int.random(in: 1...10)
    let tensorFloat: Tensor<Float> = ones(M, N)
    XCTAssertEqual(tensorFloat.shape, [M, N])
    XCTAssertEqual(tensorFloat.data.count, M*N)
    XCTAssertEqual(tensorFloat.data.reduce(Float(0), +), Float(M*N))
    
    M = Int.random(in: 1...10)
    N = Int.random(in: 1...10)
    let tensorBool: Tensor<Bool> = ones(M, N)
    XCTAssertEqual(tensorBool.shape, [M, N])
    XCTAssertEqual(tensorBool.data.count, M*N)
    let ints = tensorBool.data.map { $0 ? 1 : 0 }
    XCTAssertEqual(ints.reduce(0, +), Int(M*N))


  }
  
}
