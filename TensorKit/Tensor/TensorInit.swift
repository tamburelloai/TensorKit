//
//  TensorInit.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/8/24.
//

import Foundation

extension Tensor {
  public init(_ values: Any) {
    let (data, shape): ([T], [Int]) = Tensor.processValues(values: values)
    self.init(data: data, shape: shape, device: .cpu)
  }
  // Initializer for creating a tensor from existing data and a given shape
  init(data: [T], shape: [Int], device: DeviceType = .cpu) {
    assert(data.count == shape.reduce(1, *), "Data count does not match product of shape dimensions.")
    self.data = data
    self.shape = shape
    self.strides = Tensor.calculateStrides(for: shape)
    self.device = device
  }
  
  // Initializer for creating a tensor from an array of Double, converting to Float
  init(data: [Double], shape: [Int], device: DeviceType = .cpu) where T == Float {
    assert(data.count == shape.reduce(1, *), "Data count does not match product of shape dimensions.")
    self.data = data.map { Float($0) }
    self.shape = shape
    self.strides = Tensor.calculateStrides(for: shape)
    self.device = device
  }
}
