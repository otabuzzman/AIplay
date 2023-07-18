import PlaygroundTester
import Foundation

@objcMembers
final class LayerTests: TestCase {
    internal func testLayerInit() {
        _ = Layer(numberOfInputs: 15, numberOfPUnits: 10, activationFunction: { -$0 })
    }
}
