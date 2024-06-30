//
//  pow.swift
//  TensorKit
//
//  Created by Michael Tamburello on 6/21/24.
//

import Foundation

// Handle Int Tensor with Int Exponent -> Tensor<Int>
func pow(_ tensor: Tensor<Int>, _ exponent: Int) -> Tensor<Int> {
    let newData = tensor.data.map { element -> Int in
        let result = pow(Double(element), Double(exponent))
        if result.isFinite {
            return Int(result)
        } else {
            // Handle non-finite results, here returning 0 or some other appropriate fallback value
            return 0
        }
    }
    return Tensor<Int>(data: newData, shape: tensor.shape)
}


// Handle Float Tensor with Float Exponent -> Tensor<Float>
func pow(_ tensor: Tensor<Float>,  _ exponent: Float) -> Tensor<Float> {
    let newData = tensor.data.map { pow($0, exponent) }
    return Tensor<Float>(data: newData, shape: tensor.shape)
}

// Handle Float Tensor with Int Exponent -> Tensor<Float>
func pow(_ tensor: Tensor<Float>, _ exponent: Int) -> Tensor<Float> {
    let newData = tensor.data.map { pow($0, Float(exponent)) }
    return Tensor<Float>(data: newData, shape: tensor.shape)
}

// Handle Int Tensor with Float Exponent -> Tensor<Float>
func pow(_ tensor: Tensor<Int>, _ exponent: Float) -> Tensor<Float> {
    let newData = tensor.data.map { pow(Float($0), exponent) }
    return Tensor<Float>(data: newData, shape: tensor.shape)
}
