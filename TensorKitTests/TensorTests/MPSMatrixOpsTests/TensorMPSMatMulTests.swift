//
//  matMulTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/18/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorMPSMatMulTests: XCTestCase {
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
  
  func testStaticMatMul() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([
      [1, 2, 3],
      [3, 4, 5]
    ]).to(device)
    let t2: Tensor<Float> = Tensor([
      [1, 2],
      [3, 4],
      [1, 0]
    ]).to(device)
    
    let result = Tensor.matMul(t1, t2)
    XCTAssertEqual(result.nestedArray as! [[Float]], [[10.0, 10], [20, 22]])
    XCTAssertEqual(result.data, [10.0, 10.0, 20.0, 22.0])
    XCTAssertEqual(result.shape, [t1.shape[0], t2.shape[1]])
    XCTAssertEqual(result.device, device)
  }
  
  func testMatMul() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([
      [1, 2, 3],
      [3, 4, 5]
    ]).to(device)
    let t2: Tensor<Float> = Tensor([
      [1, 2],
      [3, 4],
      [1, 0]
    ]).to(device)
    
    let result = t1.matMul(t2)
    XCTAssertEqual(result.nestedArray as! [[Float]], [[10.0, 10], [20, 22]])
    XCTAssertEqual(result.data, [10.0, 10.0, 20.0, 22.0])
    XCTAssertEqual(result.shape, [t1.shape[0], t2.shape[1]])
    XCTAssertEqual(result.device, device)
  }
  
  func testMatMulIdentity() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([
      [1, 2, 3],
      [4, 5, 6]
    ]).to(device)
    let identity: Tensor<Float> = Tensor([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ]).to(device)
    
    let result = t1.matMul(identity)
    XCTAssertEqual(result.nestedArray as! [[Float]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    XCTAssertEqual(result.data, t1.data)
    XCTAssertEqual(result.shape, t1.shape)
    XCTAssertEqual(result.device, device)
  }
  
  func testMatMulZeroMatrix() {
    let device: DeviceType = .mps
    let t1: Tensor<Float> = Tensor([
      [1, 2],
      [3, 4]
    ]).to(device)
    let zeroMatrix: Tensor<Float> = Tensor([
      [0, 0],
      [0, 0]
    ]).to(device)
    
    let result = t1.matMul(zeroMatrix)
    XCTAssertEqual(result.nestedArray as! [[Float]], [[0.0, 0.0], [0.0, 0.0]])
    XCTAssertEqual(result.data, [0.0, 0.0, 0.0, 0.0])
    XCTAssertEqual(result.shape, [t1.shape[0], zeroMatrix.shape[1]])
    XCTAssertEqual(result.device, device)
  }
  
  
  func testMatMul_IntTensor_MPS() {
    let device: DeviceType = .mps
    let t1: Tensor<Int> = Tensor([
      [1, 2],
      [3, 4]
    ]).to(device)
    let t2: Tensor<Int> = Tensor([
      [5, 6],
      [7, 8]
    ]).to(device)
    
    let result = t1.matMul(t2)
    XCTAssertEqual(result.nestedArray as! [[Int]], [[19, 22], [43, 50]])
    XCTAssertEqual(result.data, [19, 22, 43, 50])
    XCTAssertEqual(result.shape, [t1.shape[0], t2.shape[1]])
    XCTAssertEqual(result.device, device)
  }
  
}

