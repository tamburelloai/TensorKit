//
//  calculateStridesTest.swift
//  ScorchTests
//
//  Created by Michael Tamburello on 6/8/24.
//

import XCTest
@testable import TensorKit

extension TensorTests {
    
    func testCalculateStrides() {
        // Test 1: Empty Shape
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: []), [])

        // Test 2: Scalar
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [1]), [1])

        // Test 3: Vector
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [5]), [1])

        // Test 4: Matrix
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [3, 4]), [4, 1])

        // Test 5: 3D Tensor<Int>
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [2, 3, 4]), [12, 4, 1])

        // Test 6: 4D Tensor<Int>
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [2, 3, 4, 5]), [60, 20, 5, 1])

        // Test 7: High Dimensionality
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [1, 2, 3, 4, 5]), [120, 60, 20, 5, 1])

        // Test 8: Single Dimension Large Size
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [100]), [1])

        // Test 9: Varying Sizes
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [1, 10, 1, 10]), [100, 10, 10, 1])

        // Test 10: Non-Uniform Shape
        XCTAssertEqual(Tensor<Int>.calculateStrides(for: [4, 2, 6, 1]), [12, 6, 1, 1])
    }
}
