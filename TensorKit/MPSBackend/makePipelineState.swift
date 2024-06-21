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
  
  public func getComputePipeline<T>(for shaderFunctionName: String, ofType: T) -> MTLComputePipelineState? {
    let finalFunctionName: String = shaderFunctionName + "_" + String(describing: ofType)
    if let pipeline = computePipelines[finalFunctionName] { return pipeline } else {
      let pipeline = makeComputePipelineState(shaderFunctionName: finalFunctionName)
      self.computePipelines[finalFunctionName] = pipeline
      return pipeline
    }
  }
}
