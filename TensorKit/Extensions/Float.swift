//
//  Float.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

// TODO: Change this.

extension Float: TensorData {
  public static var zero: Float { return 0.0 }
  
  static func sampleFromUniform(a: Float, b: Float) -> Float {
    precondition(a < b, "Lower bound 'a' must be less than upper bound 'b'")
    let randomValue = Float.random(in: 0..<1)
    return a + randomValue * (b - a)
  }
  
  static func sampleFromNormal(mu: Float, sigma: Float) -> Float {
    let u1 = Float.random(in: 0..<1)
    let u2 = Float.random(in: 0..<1)
    let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    return z0 * sigma + mu
  }
  
  static func _exp(_ x: Float) -> Float {
    return exp(x)
  }
  
  
}

