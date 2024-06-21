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
  
  
  func testDotProduct_2DInput_IntTensor_MPS() {
    let device: DeviceType = .mps
    let t1: Tensor<Int> = Tensor(data: [1, 2, 3, 4], shape: [1, 4], device: .mps)
    let t2: Tensor<Int> = Tensor(data: [4, 7, 3, 4], shape: [4, 1]).to(device)
    let result: Tensor<Int> = t1.dot(t2)
    let trueProductValue: Int = 43
    XCTAssertEqual(result.item(), trueProductValue)
    XCTAssertEqual(result.data, [trueProductValue])
    XCTAssertEqual(result.shape, [1, 1])
    XCTAssertEqual(result.device, device)
  }
  
  func testDotProduct_1DInput_IntTensor_MPS() {
    let device: DeviceType = .mps
    let t1: Tensor<Int> = Tensor(data: [1, 2, 3, 4], shape:  [4], device: .mps)
    let t2: Tensor<Int> = Tensor(data: [4, 7, 3, 4], shape: [4]).to(device)
    let result: Tensor<Int> = t1.dot(t2)
    let trueProductValue: Int = 43
    XCTAssertEqual(result.item(), trueProductValue)
    XCTAssertEqual(result.data, [trueProductValue])
    XCTAssertEqual(result.shape, [1])
    XCTAssertEqual(result.device, device)
  }
  
}

