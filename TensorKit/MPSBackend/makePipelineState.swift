//
//  makePipelineState.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation
import Metal

extension MPSBackend {
  func makeComputePipelineState(shaderFunctionName: String) -> MTLComputePipelineState? {
    guard let function = defaultLibrary.makeFunction(name: shaderFunctionName) else {
      print(errorFunctionString(shaderFunctionName: shaderFunctionName))
      return nil
    }
    do {
      return try device.makeComputePipelineState(function: function)
    } catch {
      print("Failed to create compute pipeline state: \(error)")
      return nil
    }
  }
  
  func errorFunctionString(shaderFunctionName: String) -> String {
    return "Shader function ***\(shaderFunctionName)*** not found in the default library."
  }
  
  func getComputePipeline(for shaderFunctionName: String) -> MTLComputePipelineState? {
    if let pipeline = computePipelines[shaderFunctionName] { return pipeline } else {
      let pipeline = makeComputePipelineState(shaderFunctionName: shaderFunctionName)
      self.computePipelines[shaderFunctionName] = pipeline
      return pipeline
    }
  }
}
