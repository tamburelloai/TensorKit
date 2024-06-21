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
  init?<U: TensorData>(_ value: U)

  
}

extension Int: TensorData {
  public static var zero: Int { return 0 }
  public static var one: Int { return 1 }
  public init?<U: TensorData>(_ value: U) {
          if let v = value as? Int {
              self = v
          } else if let v = value as? Float {
              self = Int(v)
          } else if let v = value as? Bool {
              self = v ? 1 : 0
          } else {
              return nil
          }
      }

}

extension Float: TensorData {
  public static var zero: Float { return Float(0) }
  public static var one: Float { return  Float(1) }
  public init?<U: TensorData>(_ value: U) {
          if let v = value as? Float {
              self = v
          } else if let v = value as? Int {
              self = Float(v)
          } else if let v = value as? Bool {
              self = v ? 1.0 : 0.0
          } else {
              return nil
          }
      }
}


extension Bool: TensorData {
  public static var zero: Bool { return false }
  public static var one: Bool { return true }
  static func +(lhs: Bool, rhs: Bool) -> Bool {
    return lhs || rhs  // OR operation
  }
  public init?<U: TensorData>(_ value: U) {
          if let v = value as? Bool {
              self = v
          } else if let v = value as? Int {
              self = v != 0
          } else if let v = value as? Float {
              self = v != 0.0
          } else {
              return nil
          }
      }
}
