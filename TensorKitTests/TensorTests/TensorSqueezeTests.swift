//
//  TensorSqueezeTests.swift
//  TensorKitTests
//
//  Created by Michael Tamburello on 6/20/24.
//

import Foundation
import XCTest
@testable import TensorKit

final class TensorSqueezeTests: XCTestCase {
  func testSimpleSqueeze() {
    var x: Tensor = rand(1, 10, 1)
    let y: Tensor = x.squeeze(0)
    XCTAssertEqual(y.shape, [10, 1])
    let z: Tensor = y.squeeze(-1)
    XCTAssertEqual(x.shape, [1, 10, 1])
    XCTAssertEqual(y.shape, [10, 1])
    XCTAssertEqual(z.shape, [10])
  }
  
  func testSimpleSqueezeInplace() {
    var x: Tensor = rand(1, 10, 1)
    x.squeeze(0, inplace: true)
    XCTAssertEqual(x.shape, [10, 1])
    x.squeeze(-1, inplace: true)
    XCTAssertEqual(x.shape, [10])
  }
}
