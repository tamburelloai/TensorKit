//
//  processValues.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation



extension Tensor where T: TensorData {
  static func processValues(values: Any) -> ([T], [Int]) {
    var shape = [Int]()
    var flatValues = [T]()
    func flatten(_ values: Any, currentShape: inout [Int]) {
      if let array = values as? [Any] {
        if let firstSubArray = array.first as? [Any] {
          let expectedSize = firstSubArray.count
          for element in array {
            guard let subArray = element as? [Any], subArray.count == expectedSize else {
              fatalError("subarrays at the same level must be of the same size")
            }
          }
        }
        currentShape.append(array.count)
        for element in array {
          flatten(element, currentShape: &currentShape)
        }
      } else if let value = values as? T {
        flatValues.append(value)
      } else if let value = values as? Double {
        if let finalValue = Float(value) as? T {
          flatValues.append(finalValue)
        } else {
          fatalError("Unsupported type: Failed double->float conversion")
        }
      } else {
        fatalError("Unsupported type in array")
      }
    }
    flatten(values, currentShape: &shape)
    var actualShape = [Int]()
    var currentLevel: Any = values
    while let array = currentLevel as? [Any] {
      actualShape.append(array.count)
      currentLevel = array.first ?? []
    }
    return (flatValues, actualShape)
  }
}
