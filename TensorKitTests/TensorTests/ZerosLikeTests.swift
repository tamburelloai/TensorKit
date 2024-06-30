//
//  ZerosLikeTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/20/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class ZerosLikeTests: XCTestCase {
  func testZerosLike() {
    let x: Tensor<Float> = ones(shape: [10, 20])
    let zeros1: Tensor<Float> = zerosLike(x)
    let zeros2: Tensor<Float> = zeros(like: x)
    XCTAssertEqual(zeros1.data, zeros2.data)
    XCTAssertEqual(zeros1.data.count, x.shape.reduce(1, *))
    XCTAssertEqual(zeros1.shape, zeros2.shape)
    XCTAssertEqual(zeros1.device, zeros2.device)
    XCTAssertEqual(zeros1.nestedArray as! [[Float]], zeros2.nestedArray as! [[Float]])
    XCTAssertEqual(x.shape, zeros1.shape)
    XCTAssertEqual(x.device, zeros1.device)
  }
}
