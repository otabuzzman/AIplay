import SwiftUI

struct MYONNView: View {
    @StateObject private var dataset = MNISTDataset(in: getAppFolder())
    
    var body: some View {
        NetworkView(config: defaultConfig, dataset: dataset)
    }
}

// NYONN sample configuration
//   usage: GenericFactory.create(NetworkFactory(), defaultConfig)
let defaultConfig: NetworkConfig = (
    layersWithSizes: [784, 100, 10], activationFunctions: [.sigmoid, .sigmoid], learningRate: 0.3
)

// specialized NYONN factory
//   usage: GenericFactory.create(DefaultFactory(), nil)
struct DefaultFactory: AbstractFactory {
    func create(_ config: Never?) -> Network? {
        Network([
            Layer(numberOfInputs: 784, numberOfPUnits: 100, activationFunction: .sigmoid),
            Layer(numberOfInputs: 100, numberOfPUnits: 10, activationFunction: .sigmoid)
        ], alpha: 0.3)
    }
}
