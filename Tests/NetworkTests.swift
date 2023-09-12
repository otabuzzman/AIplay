import PlaygroundTester
import Foundation

@objcMembers
final class NetworkTests: TestCase {
    internal func testNetworkInit() {
        _ = NetworkView.NetworkViewModel(GenericFactory.create(NetworkFactory(), ([3, 2, 1], [.identity, .identity], 0.5))!, MNISTDataset(in: nil))
    }
    
    internal func testLayerInit() {
        _ = Layer(numberOfInputs: 47, numberOfPUnits: 11)
        _ = Layer(numberOfInputs: 47, numberOfPUnits: 11, activationFunction: .sigmoid)
        _ = Layer(numberOfInputs: 47, numberOfPUnits: 11, weights: Matrix(rows: 8, columns: 15))
    }
}
