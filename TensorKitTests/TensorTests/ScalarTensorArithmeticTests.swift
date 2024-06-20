//
//  ScalarTensorCPUArithmeticTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/20/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class ScalarTensorCPUAdditionTests: XCTestCase {
  func testIntScalarTensorAddition() {
    let device: DeviceType = .cpu
    let scalar: Int = 2
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar + tensor
    let resultSwitched: Tensor<Float> = tensor + scalar
    XCTAssertEqual(result.data, resultSwitched.data)
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) + inputData[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testFloatScalarTensorAddition() {
    let device: DeviceType = .cpu
    let scalar: Float = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar + tensor
    let resultSwitched: Tensor<Float> = tensor + scalar
    XCTAssertEqual(result.data, resultSwitched.data)
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) + inputData[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testDoubleScalarTensorAddition() {
    let device: DeviceType = .cpu
    let scalar: Double = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar + tensor
    let resultSwitched: Tensor<Float> = tensor + scalar
    XCTAssertEqual(result.data, resultSwitched.data)
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) + inputData[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
}

final class ScalarTensorCPUSubtractionTests: XCTestCase {
  func testIntScalarTensorSubtraction() {
    let device: DeviceType = .cpu
    let scalar: Int = 2
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar - tensor
    let resultSwitched: Tensor<Float> = tensor - scalar
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) - inputData[i], accuracy: 0.999)
      XCTAssertEqual(resultSwitched.data[i], inputData[i] - Float(scalar), accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testFloatScalarTensorSubtraction() {
    let device: DeviceType = .cpu
    let scalar: Float = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    
    let result: Tensor<Float> = scalar - tensor
    let resultSwitched: Tensor<Float> = tensor - scalar
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) - inputData[i], accuracy: 0.999)
      XCTAssertEqual(resultSwitched.data[i], inputData[i] - Float(scalar), accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testDoubleScalarTensorSubtraction() {
    let device: DeviceType = .cpu
    let scalar: Double = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    
    let result: Tensor<Float> = scalar - tensor
    let resultSwitched: Tensor<Float> = tensor - scalar
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) - inputData[i], accuracy: 0.999)
      XCTAssertEqual(resultSwitched.data[i], inputData[i] - Float(scalar), accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
}

final class ScalarTensorCPUMultiplicationTests: XCTestCase {
  func testIntScalarTensorAddition() {
    let device: DeviceType = .cpu
    let scalar: Int = 2
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar * tensor
    let resultSwitched: Tensor<Float> = tensor * scalar
    XCTAssertEqual(result.data, resultSwitched.data)
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) * inputData[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testFloatScalarTensorAddition() {
    let device: DeviceType = .cpu
    let scalar: Float = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar * tensor
    let resultSwitched: Tensor<Float> = tensor * scalar
    XCTAssertEqual(result.data, resultSwitched.data)
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) * inputData[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testDoubleScalarTensorAddition() {
    let device: DeviceType = .cpu
    let scalar: Double = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar * tensor
    let resultSwitched: Tensor<Float> = tensor * scalar
    XCTAssertEqual(result.data, resultSwitched.data)
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) * inputData[i], accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
}


final class ScalarTensorCPUDivisionTests: XCTestCase {
  func testIntScalarTensor() {
    let device: DeviceType = .cpu
    let scalar: Int = 2
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    let result: Tensor<Float> = scalar / tensor
    let resultSwitched: Tensor<Float> = tensor / scalar
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) / inputData[i], accuracy: 0.999)
      XCTAssertEqual(resultSwitched.data[i], inputData[i] / Float(scalar), accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testFloatScalarTensor() {
    let device: DeviceType = .cpu
    let scalar: Float = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    
    let result: Tensor<Float> = scalar / tensor
    let resultSwitched: Tensor<Float> = tensor / scalar
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) / inputData[i], accuracy: 0.999)
      XCTAssertEqual(resultSwitched.data[i], inputData[i] / Float(scalar), accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
  
  func testDoubleScalarTensor() {
    let device: DeviceType = .cpu
    let scalar: Double = 2.0
    let inputData: [Float] = [1.0, 2, 3, 4, 5]
    let tensor: Tensor<Float> = Tensor(data: inputData, shape: [1, 5], device: device)
    
    let result: Tensor<Float> = scalar / tensor
    let resultSwitched: Tensor<Float> = tensor / scalar
    for i in (0..<tensor.data.count) {
      XCTAssertEqual(result.data[i], Float(scalar) / inputData[i], accuracy: 0.999)
      XCTAssertEqual(resultSwitched.data[i], inputData[i] / Float(scalar), accuracy: 0.999)
    }
    XCTAssertEqual(tensor.device, result.device)
  }
}


