//
//  MPSBackendError.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/12/24.
//

import Foundation

enum MPSBackendError: Error {
  case metalNotSupported
  case commandQueueCreationFailed
  case defaultLibraryNotFound
  case computePipelineStateCreationFailed(Error)
}
