//
//  item.swift
//  Scorch
//
//  Created by Michael Tamburello on 6/14/24.
//

import Foundation

extension Tensor where T: TensorData & Numeric {
  func item() -> T {
    assert(self.data.count == 1)
    return self.data.first!
  }
}
