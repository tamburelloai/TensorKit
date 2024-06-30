//
//  Tensor.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/6/24.
//



import Foundation

public struct Tensor<T:TensorData>: Hashable {
  var id: String
  public var data: [T]
  public var shape: [Int]
  public var strides: [Int]
  public var device: DeviceType
  
  
  /// Initializer that allows for intuitive tensor initialization from a nested array.
  public init(_ nestedArray: Any, requiresGrad: Bool = false) {
    let array = nestedArray as! [Any]
    if (!array.isEmpty) {
      let (data, shape): ([T], [Int]) = Tensor.processValues(values: nestedArray)
      self.init(data: data, shape: shape, device: .cpu, requiresGrad: requiresGrad)
    } else {
      /// empty tensor init support
      self.init(data: [], shape: [], device: .cpu, requiresGrad: requiresGrad)
    }
  }
  
  // Initializer for creating a tensor from existing data and a given shape
  public init(data: [T], shape: [Int], strides: [Int] = [], device: DeviceType = .cpu, requiresGrad: Bool = false) {
    assert(shape.isEmpty || data.count == shape.reduce(1, *), "Data count does not match product of shape dimensions.")
    self.id = UUID().uuidString
    self.data = data
    self.shape = shape
    self.strides =  strides.isEmpty ? Tensor.calculateStrides(for: shape) : strides
    self.device = device
    
  }
  
  // Initializer for creating a tensor from an array of Double, converting to Float
  public init(data: [Double], shape: [Int], device: DeviceType = .cpu, requiresGrad: Bool = false) where T == Float {
    self.init(data: data.map {Float($0)}, shape: shape)
  }
  
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
      } else if let value = values as? Int {
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
  
  static func calculateStrides(for shape: [Int]) -> [Int] {
    var strides = Array(repeating: 1, count: shape.count)
    for i in stride(from: shape.count - 2, through: 0, by: -1) {
      strides[i] = strides[i + 1] * shape[i + 1]
    }
    return strides
  }
  
  /// Returns nested array representation of a `Tensor`
  var nestedArray: Any {
    return createNestedArray(dim: 0, index: [])
  }
  
  private func createNestedArray(dim: Int, index: [Int]) -> Any {
    if dim == shape.count - 1 {
      return (0..<shape[dim]).map { element(at: index + [$0]) }
    } else {
      return (0..<shape[dim]).map { createNestedArray(dim: dim + 1, index: index + [$0]) }
    }
  }
  
  /// Returns the dimension sizes representing the tensor object
  var ndim: Int { return self.shape.count }
  
  /* Subscript for indexing the tensor with **multiple** dimensions
   Example: Getting the (i,j) value - `Tensor[i, j]` => value at index (i, j) */
  public subscript(indices: Int...) -> T {
    get {
      let index = calculateIndex(indices: indices)
      return data[index]
    }
    set {
      let index = calculateIndex(indices: indices)
      data[index] = newValue
    }
  }
  
  /// Function to return an element at a given index.
  public func element(at index: [Int]) -> T {
    var flatIndex = 0
    for (i, idx) in index.enumerated() {
      flatIndex += idx * strides[i]
    }
    return data[flatIndex]
  }
  
  /// Helper function to return index location of value in the underlying (flat) data array
  private func calculateIndex(indices: [Int]) -> Int {
    assert(indices.count == shape.count, "Index count does not match shape dimensions.")
    return indices.enumerated().reduce(0) { $0 + $1.element * strides[$1.offset] }
  }
  
  /// A convenient function that changes which processing unit to use.
  /// for any subsequent operations
  public func to(_ device: DeviceType) -> Tensor {
    var newTensor = self
    newTensor.device = device
    return newTensor
  }
  
  /// Function that will turn tensor into a different data type
  func astype<U: TensorData>(_ type: TensorType) -> Tensor<U> {
    switch type {
    case .bool:
      guard let newData = data.map({ Bool($0) }) as? [U] else { fatalError("Conversion failed") }
      return Tensor<U>(newData)
    case .int:
      guard let newData = data.map({ Int($0) }) as? [U] else { fatalError("Conversion failed") }
      return Tensor<U>(newData)
    case .float:
      guard let newData = data.map({ Float($0) }) as? [U] else { fatalError("Conversion failed") }
      return Tensor<U>(newData)
    }
  }
  
  /// String representation of a `Tensor` object
  public var description: String {
    return "\(self.nestedArray)"
  }
  
  /// Returns the single item in a `Tensor` if there is only one item
  /// in the underlying `.data` array
  func item() -> T {
    assert(self.data.count == 1)
    return self.data.first!
  }
  
  /// required so that `Tensor` to conform to the `Tensor` protocol
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    return lhs.id == rhs.id
  }
  
  /// required so that `Tensor` to conform to the `Tensor` protocol
  public func hash(into hasher: inout Hasher) {
    hasher.combine(id)
  }
}

