//
//  TensorTestNDimComputedVar.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/21/24.
//

import XCTest
@testable import TensorKit

final class TensorTestNDimComputedVar: XCTestCase {
  
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
  
  func testNDimComputedVar() {
    let t1 = rand(5, 10)
    let t2 = rand(4, 5, 10)
    let t3 = rand(1, 1, 5, 10)
    let t4 = rand(10)
    
    XCTAssertEqual(t1.ndim, 2)
    XCTAssertEqual(t2.ndim, 3)
    XCTAssertEqual(t3.ndim, 4)
    XCTAssertEqual(t4.ndim, 1)
  }
  
}
