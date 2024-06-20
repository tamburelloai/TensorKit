//
//  TensorInitTests.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/6/24.
//

import XCTest
@testable import TensorKit

final class TensorTests: XCTestCase {
  
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
  
  func testElement() {
    let x: Tensor<Float> = Tensor(data: [1.0, 2, 3, 4], shape: [4])
    XCTAssertEqual(x.element(at: [0]), Float(1.0))
    XCTAssertEqual(x.element(at: [1]), Float(2.0))
    XCTAssertEqual(x.element(at: [2]), Float(3.0))
    XCTAssertEqual(x.element(at: [3]), Float(4.0))
  }
  
  func testElement2D() {
    let x: Tensor<Float> = Tensor([
    [1.0, 2, 3, 4],
    [1.0, 2, 3, 4],
    [1.0, 2, 3, 4]])
    XCTAssertEqual(x.element(at: [0, 1]), Float(2.0))
    XCTAssertEqual(x.element(at: [1, 3]), Float(4.0))
    XCTAssertEqual(x.element(at: [1, 1]), Float(2.0))
    XCTAssertEqual(x.element(at: [2, 1]), Float(2.0))
  }
  
  func testIndexSetter() {
    var x: Tensor<Float> = Tensor([
    [1.0, 2, 3, 4],
    [1.0, 2, 3, 4],
    [1.0, 2, 3, 4]])
    x[0, 0] = Float(69.0)
    x[2, 2] = Float(14.0)

    XCTAssertEqual(x.element(at: [0, 0]), Float(69.0))
    XCTAssertEqual(x.element(at: [1, 3]), Float(4.0))
    XCTAssertEqual(x.element(at: [1, 1]), Float(2.0))
    XCTAssertEqual(x.element(at: [2, 2]), Float(14.0))
  }
  
  
}
