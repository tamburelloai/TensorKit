//
//  MPSBackend.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation
import Metal

/// `MPSBackend` is a singleton class designed to manage Metal-based GPU computations
/// efficiently within your application. It abstracts the complexities of direct Metal API interactions
/// and provides a streamlined interface for performing high-performance compute tasks.
/// - **Properties**:
///   - `device (MTLDevice)`                        : main interface to the GPU - will call methods on this to create GPU-specific objects
///   - `queue (MTLCommandQueue)`             : Used to create, submit, and schedule command buffers to a specific GPU device to run the commands within those buffers
///   - `shaders (MTLLibrary)`                    : A collection of metal shader functions



public class MPSBackend {
  public var device: MTLDevice!
  public var commandQueue: MTLCommandQueue!
  public var defaultLibrary: MTLLibrary!
  public var computePipelineState: MTLComputePipelineState!
  public var initializationError: Error?
  public var computePipelines: [String: MTLComputePipelineState] = [:]
  public static let shared = MPSBackend() // singleton instance
  
  
  private init() {
    do {
      try setupMetal()
    } catch {
      print("Failed to initialize MetalBackend: \(error)")
      initializationError = error
    }
  }
  
  private func setupMetal() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MPSBackendError.metalNotSupported
    }
    self.device = device
    
    guard let commandQueue = device.makeCommandQueue() else {
      throw MPSBackendError.commandQueueCreationFailed
    }
    self.commandQueue = commandQueue
    
    // Load the default library
    let frameworkBundle = Bundle(for: MPSBackend.self)
    guard let defaultLibrary = try? device.makeDefaultLibrary(bundle: frameworkBundle) else {
      fatalError("Could not load default library from specified bundle")
    }
    self.defaultLibrary = defaultLibrary
  }
  
  public func createBufferFromTensor<T:TensorData&Numeric>(_ tensor: Tensor<T>) -> MTLBuffer {
    let dataSize: Int = tensor.data.count * MemoryLayout<T>.size
    return device.makeBuffer(bytes: tensor.data, length: dataSize, options: .storageModeShared)!
  }
}

extension MPSBackend {
  func dot<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.shape == rhs.shape, "MPS.dot requires identical tensor shapes, got \(lhs.shape), \(rhs.shape)")
    var result: Tensor<T> = zeros(1)
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "dot", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    commandEncoder.setComputePipelineState(computePipelineState)
    commandEncoder.setBuffer(createBufferFromTensor(lhs), offset: 0, index: 0)
    commandEncoder.setBuffer(createBufferFromTensor(rhs), offset: 0, index: 1)
    
    let resultBuffer: MTLBuffer = createBufferFromTensor(result)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadCount: Int = result.data.count
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1) // This should divide the grid size without remainder.
    let numThreadgroups = MTLSize(width: (threadCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
  func matMul<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "matMul", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    commandEncoder.setComputePipelineState(computePipelineState)
    commandEncoder.setBuffer(createBufferFromTensor(lhs), offset: 0, index: 0)
    commandEncoder.setBuffer(createBufferFromTensor(rhs), offset: 0, index: 1)
    
    let resultShape: [Int] = [lhs.shape[0], rhs.shape[1]]
    var result: Tensor<T> = zeros(shape: resultShape)
    let resultBuffer: MTLBuffer = createBufferFromTensor(result)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    // M, K, N setup
    let M = lhs.shape[0]
    let K = lhs.shape[1]
    let N = rhs.shape[1] // outer dim (N)
    var m = UInt32(M)
    var k = UInt32(K)
    var n = UInt32(N)
    // Create buffers for the dimensions
    commandEncoder.setBuffer(device.makeBuffer(bytes: &m, length: MemoryLayout<UInt32>.size, options: .storageModeShared), offset: 0, index: 3)
    commandEncoder.setBuffer(device.makeBuffer(bytes: &k, length: MemoryLayout<UInt32>.size, options: .storageModeShared), offset: 0, index: 4)
    commandEncoder.setBuffer(device.makeBuffer(bytes: &n, length: MemoryLayout<UInt32>.size, options: .storageModeShared), offset: 0, index: 5)
    
    
    let threadCount: Int = result.data.count
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1) // This should divide the grid size without remainder.
    let numThreadgroups = MTLSize(width: (threadCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let resultPointer = resultBuffer.contents().bindMemory(to: T.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
  func tiledMatMul<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "tiledMatMul", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    let M = lhs.shape[0]
    let K = lhs.shape[1]
    let N = rhs.shape[1]
    commandEncoder.setComputePipelineState(computePipelineState)
    let bufferSizeA = M * K * MemoryLayout<T>.size
    let bufferSizeB = K * N * MemoryLayout<T>.size
    let bufferSizeC = M * N * MemoryLayout<T>.size
    let bufferA = device.makeBuffer(bytes: lhs.data, length: bufferSizeA, options: .storageModeShared)
    let bufferB = device.makeBuffer(bytes: rhs.data, length: bufferSizeB, options: .storageModeShared)
    let bufferC = device.makeBuffer(length: bufferSizeC, options: .storageModeShared)
    let dimensions = [UInt32(M), UInt32(N), UInt32(K)]
    let bufferDimensions = device.makeBuffer(bytes: dimensions, length: dimensions.count * MemoryLayout<UInt32>.size, options: .storageModeShared)
    commandEncoder.setBuffer(bufferA, offset: 0, index: 0)
    commandEncoder.setBuffer(bufferB, offset: 0, index: 1)
    commandEncoder.setBuffer(bufferC, offset: 0, index: 2)
    commandEncoder.setBuffer(bufferDimensions, offset: 0, index: 3)
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadgroupsPerGrid = MTLSize(width: (N + 15) / 16, height: (M + 15) / 16, depth: 1)
    commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultShape: [Int] = [M, N]
    var result: Tensor<T> = zeros(shape: resultShape)
    let threadCount: Int = result.data.count
    let resultPointer = bufferC?.contents().bindMemory(to: T.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
  func batchMatrixMultiply<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.shape == rhs.shape, "MPS.batchMatrixMultiply requires identical tensor shapes, got \(lhs.shape), \(rhs.shape)")
    var result: Tensor<T> = zeros(shape: lhs.shape)
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "elementwiseAddition", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    commandEncoder.setComputePipelineState(computePipelineState)
    commandEncoder.setBuffer(createBufferFromTensor(lhs), offset: 0, index: 0)
    commandEncoder.setBuffer(createBufferFromTensor(rhs), offset: 0, index: 1)
    
    let resultBuffer: MTLBuffer = createBufferFromTensor(result)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadCount: Int = result.data.count
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1) // This should divide the grid size without remainder.
    let numThreadgroups = MTLSize(width: (threadCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
  
}


//Elementwise MPS Operations
extension MPSBackend {
  func elementwiseOperation<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>, _ operation: OperationType) -> Tensor<T> {
    switch operation {
    case .add: return self.elementwiseAddition(lhs, rhs)
    case .subtract: return self.elementwiseSubtraction(lhs, rhs)
    case .multiply: return self.elementwiseMultiplication(lhs, rhs)
    case .divide: return self.elementwiseDivision(lhs, rhs)
    }
  }
  
  
  func elementwiseAddition<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.shape == rhs.shape, "elementwiseAddition requires identical tensor shapes, got \(lhs.shape), \(rhs.shape)")
    var result: Tensor<T> = zeros(shape: lhs.shape)
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "elementwiseAddition", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    commandEncoder.setComputePipelineState(computePipelineState)
    commandEncoder.setBuffer(createBufferFromTensor(lhs), offset: 0, index: 0)
    commandEncoder.setBuffer(createBufferFromTensor(rhs), offset: 0, index: 1)
    
    let resultBuffer: MTLBuffer = createBufferFromTensor(result)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadCount: Int = result.data.count
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1) // This should divide the grid size without remainder.
    let numThreadgroups = MTLSize(width: (threadCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
  func elementwiseSubtraction<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.shape == rhs.shape, "elementwiseAddition requires identical tensor shapes, got \(lhs.shape), \(rhs.shape)")
    var result: Tensor<T> = zeros(shape: lhs.shape)
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "elementwiseSubtraction", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    commandEncoder.setComputePipelineState(computePipelineState)
    commandEncoder.setBuffer(createBufferFromTensor(lhs), offset: 0, index: 0)
    commandEncoder.setBuffer(createBufferFromTensor(rhs), offset: 0, index: 1)
    
    let resultBuffer: MTLBuffer = createBufferFromTensor(result)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadCount: Int = result.data.count
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1) // This should divide the grid size without remainder.
    let numThreadgroups = MTLSize(width: (threadCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
  func elementwiseMultiplication<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.shape == rhs.shape, "elementwiseAddition requires identical tensor shapes, got \(lhs.shape), \(rhs.shape)")
    var result: Tensor<T> = zeros(shape: lhs.shape)
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "elementwiseMultiplication", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    commandEncoder.setComputePipelineState(computePipelineState)
    commandEncoder.setBuffer(createBufferFromTensor(lhs), offset: 0, index: 0)
    commandEncoder.setBuffer(createBufferFromTensor(rhs), offset: 0, index: 1)
    
    let resultBuffer: MTLBuffer = createBufferFromTensor(result)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadCount: Int = result.data.count
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1) // This should divide the grid size without remainder.
    let numThreadgroups = MTLSize(width: (threadCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
  func elementwiseDivision<T:TensorData&Numeric>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    assert(lhs.shape == rhs.shape, "elementwiseAddition requires identical tensor shapes, got \(lhs.shape), \(rhs.shape)")
    var result: Tensor<T> = zeros(shape: lhs.shape)
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
          let computePipelineState = getComputePipeline(for: "elementwiseDivision", ofType: T.self) else {
      fatalError("Failed to create command buffer or command encoder")
    }
    commandEncoder.setComputePipelineState(computePipelineState)
    commandEncoder.setBuffer(createBufferFromTensor(lhs), offset: 0, index: 0)
    commandEncoder.setBuffer(createBufferFromTensor(rhs), offset: 0, index: 1)
    
    let resultBuffer: MTLBuffer = createBufferFromTensor(result)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
    
    let threadCount: Int = result.data.count
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1) // This should divide the grid size without remainder.
    let numThreadgroups = MTLSize(width: (threadCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: threadCount)
    result.data = Array(UnsafeBufferPointer(start: resultPointer, count: threadCount)) as! [T]
    result.device = lhs.device
    return result
  }
  
}

// Broadcast MPS Operations
extension MPSBackend {
  func broadcastOperation<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>, _ operation: OperationType) -> Tensor<T> {
    switch operation {
    case .add: return self._broadcastAdd(lhs, rhs)
    case .subtract: return self._broadcastSubtract(lhs, rhs)
    case .multiply: return self._broadcastMultiply(lhs, rhs)
    case .divide: return self._broadcastDivide(lhs, rhs)
    }
  }
  
  func _broadcastAdd<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] + rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
  
  func _broadcastSubtract<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] - rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
  
  func _broadcastMultiply<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] * rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
  
  func _broadcastDivide<T:TensorData&Numeric&FloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    let resultShape: [Int] = calculateBroadcastShape(lhs.shape, rhs.shape)
    let resultSize: Int = productOfDimensions(resultShape)
    var resultFlatArray: [T] = Array(repeating: T.zero, count: resultSize)
    let paddedShapeA = padShapeWithOnes(shape: lhs.shape, maxDims: resultShape.count)
    let paddedShapeB = padShapeWithOnes(shape: rhs.shape, maxDims: resultShape.count)
    
    for idx in 0..<resultSize {
      let multiDimIndex: [Int] = indexToMultiDim(idx, resultShape)
      let indexA = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeA[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeA
      )
      let indexB = multiDimToIndex(
        multiDimIndex.enumerated().map { paddedShapeB[$0.offset] == 1 ? 0 : $0.element },
        paddedShapeB
      )
      resultFlatArray[idx] = lhs.data[indexA] / rhs.data[indexB]
    }
    return Tensor(data: resultFlatArray, shape: resultShape)
  }
  
}
