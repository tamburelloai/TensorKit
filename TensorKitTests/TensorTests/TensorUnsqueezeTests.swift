//
//  TensorUnsqueezeTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/18/24.
//


import Foundation
import XCTest
@testable import TensorKit

final class TensorUnsqueezeTests: XCTestCase {
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
  
  func testUnsqueezeFirst() {
    let device: DeviceType = .cpu
    let t: Tensor<Float> = Tensor([
      [1, 2, 3],
      [4, 5, 6]
    ]).to(device)
    
    let result = t.unsqueeze(0)
    XCTAssertEqual(result.shape, [1, 2, 3])
    XCTAssertEqual(result.data, t.data)
    XCTAssertEqual(result.device, device)
    
    let result2 = result.unsqueeze(0)
    XCTAssertEqual(result2.shape, [1, 1, 2, 3])
    XCTAssertEqual(result2.data, t.data)
    XCTAssertEqual(result2.device, device)
    
  }
  
  func testUnsqueezeLast() {
    let device: DeviceType = .cpu
    let t: Tensor<Float> = Tensor([
      [1, 2, 3],
      [4, 5, 6]
    ]).to(device)
    
    let result = t.unsqueeze(t.shape.count)
    XCTAssertEqual(result.shape, [2, 3, 1])
    XCTAssertEqual(result.data, t.data)
    XCTAssertEqual(result.device, device)
    let result2 = result.unsqueeze(-1)
    XCTAssertEqual(result2.shape, [2, 3, 1, 1])
    XCTAssertEqual(result2.data, t.data)
    XCTAssertEqual(result2.device, device)
  }
  
  func testUnsqueezeMiddle() {
    let device: DeviceType = .cpu
    let t: Tensor<Float> = Tensor([
      [1, 2, 3],
      [4, 5, 6]
    ]).to(device)
    
    let result = t.unsqueeze(1)
    XCTAssertEqual(result.shape, [2, 1, 3])
    XCTAssertEqual(result.data, t.data)
    XCTAssertEqual(result.device, device)
  }
  
  func testUnsqueezeMiddleAgain() {
    let device: DeviceType = .cpu
    let t: Tensor<Float> = Tensor([
      [1, 2, 3],
      [4, 5, 6]
    ]).to(device)
    
    var result = t.unsqueeze(1)
    result = result.unsqueeze(2)
    XCTAssertEqual(result.shape, [2, 1, 1, 3])
    XCTAssertEqual(result.data, t.data)
    XCTAssertEqual(result.device, device)
  }
  
  
  //TODO: implement init empty tensor and then include this test
//  func testUnsqueezeEmpty() {
//    let device: DeviceType = .cpu
//    let t: Tensor<Float> = Tensor([]).to(device)
//    let result = t.unsqueeze(0)
//    XCTAssertEqual(result.shape, [1])
//    XCTAssertEqual(result.data, t.data)
//    XCTAssertEqual(result.device, device)
//  }
  
  
}
