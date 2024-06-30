//
//  TensorType.swift
//  TensorKit
//
//  Created by Michael Tamburello on 6/21/24.
//

import Foundation


enum TensorType {
  case bool
  case int
  case float
  
  func typeReference(_ type: TensorType) -> Any.Type {
    switch type {
    case .bool: return Bool.self
    case .int:  return Float.self
    case .float:  return Float.self
    }
  }
  
}
