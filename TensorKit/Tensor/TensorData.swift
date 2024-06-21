//
//  TensorData.swift
//  TensorKit
//
//  Created by Michael Tamburello on 6/7/24.
//

import Foundation

public protocol TensorData {
  static var zero: Self { get }
  static var one: Self { get }
}

extension Float: TensorData {
  public static var zero: Float { return Float(0) }
  public static var one: Float { return  Float(1) }
}

extension Int: TensorData {
  public static var zero: Int { return 0 }
  public static var one: Int { return 1 }
}

extension Bool: TensorData {
  public static var zero: Bool { return false }
  public static var one: Bool { return true }
  static func +(lhs: Bool, rhs: Bool) -> Bool {
    return lhs || rhs  // OR operation
  }
}






