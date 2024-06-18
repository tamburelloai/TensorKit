//
//  TensorMPSDotProductTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/18/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorMPSDotProductTests: XCTestCase {
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
  
  func testStaticDotProduct1D() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([1, 2, 3]).to(device)
    let t2: Tensor<Float> = Tensor([1, 2, 3]).to(device)
    let result = Tensor.dot(t1, t2)
    XCTAssertEqual(result.data, [14.0])
    XCTAssertEqual(result.shape, [1])
    XCTAssertEqual(result.device, device)
  }
  
  func testStaticDotProduct2D() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([[1, 2, 3]]).to(device)
    let t2: Tensor<Float> = Tensor([[1], [2], [3]]).to(device)
    let result = Tensor.dot(t1, t2)
    XCTAssertEqual(result.data, [14.0])
    XCTAssertEqual(result.shape, [1, 1])
    XCTAssertEqual(result.shape.count, t1.shape.count)
    XCTAssertEqual(result.shape.count, t2.shape.count)
    XCTAssertEqual(result.device, device)
  }
  
  func testDotProduct1D() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([1, 2, 3]).to(device)
    let t2: Tensor<Float> = Tensor([1, 2, 3]).to(device)
    let result = t1.dot(t2)
    XCTAssertEqual(result.data, [14.0])
    XCTAssertEqual(result.shape, [1])
    XCTAssertEqual(result.device, device)
  }
  
  func testDotProduct2D() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([[1, 2, 3]]).to(device)
    let t2: Tensor<Float> = Tensor([[1], [2], [3]]).to(device)
    let result = t1.dot(t2)
    XCTAssertEqual(result.data, [14.0])
    XCTAssertEqual(result.shape, [1, 1])
    XCTAssertEqual(result.shape.count, t1.shape.count)
    XCTAssertEqual(result.shape.count, t2.shape.count)
    XCTAssertEqual(result.device, device)
  }
  
  //TODO: figure out how to have both assertions and tests to ensure those
  // assertions are failed (resulting in passing test when that is supposed to be the case
  // without having to use optionals, if let and all of that.
//  func testStaticDotProduct2DWrongShapes() {
//    let device: DeviceType = .mps
//    let t1: Tensor<Float> = Tensor([[1, 2, 3]]).to(device)
//    let t2: Tensor<Float> = Tensor([[1, 2, 3]]).to(device)
//    XCTAssertThrowsError(try Tensor.dot(t1, t2))
//  }
  
 
  
  
  //TODO: fix dot to allow for Int type
  //  func testDotDifferentDataTypes() {
  //      let device: DeviceType = .mps
  //      let t1: Tensor<Int> = Tensor([
  //        [1, 2],
  //        [3, 4]
  //      ]).to(device)
  //      let t2: Tensor<Int> = Tensor([
  //        [5, 6],
  //        [7, 8]
  //      ]).to(device)
  //
  //      let result = t1.matMul(t2)
  //      XCTAssertEqual(result.nestedArray() as! [[Int]], [[19, 22], [43, 50]])
  //      XCTAssertEqual(result.data, [19, 22, 43, 50])
  //      XCTAssertEqual(result.shape, [t1.shape[0], t2.shape[1]])
  //      XCTAssertEqual(result.device, device)
  //  }
  
}

