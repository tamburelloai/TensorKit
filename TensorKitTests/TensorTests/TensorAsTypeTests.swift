//
//  TensorAsTypeTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/20/24.
//

import XCTest
import Foundation
@testable import TensorKit

final class TensorAsTypeTests: XCTestCase {
  
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
  
  func testFloatToInt() {
    let x: Tensor<Float> = Tensor([1.0, 2.0, 3.0, 4.0])
    let y: Tensor<Int> = x.astype(.int)
    XCTAssertEqual(y.data, [1, 2, 3, 4])
    let a: Tensor<Int> = Tensor([1, 2])
    let b: Tensor<Float> = a.astype(.float)
    XCTAssertEqual(b.data, [1.0, 2.0])

  }
  
  
}
