import SwiftUI

struct MYONN: View {
    @State private var folderPickerShow = getAppFolder() == nil
    
    @StateObject var viewModel = MYONNViewModel()
    
    @StateObject var mnist = MNISTViewModel(in: getAppFolder())
    // specialized MYONN factory
    @StateObject var network = NetworkViewModel(GenericFactory.create(MYONNFactory(), nil)!)
    // Network factory with MYONN configuration
    // @StateObject var network = NetworkViewModel(GenericFactory.create(NetworkFactory(), MYONNConfig)!)
    
    var body: some View {
        HStack {
            MNISTView(viewModel: mnist)
            NetworkView(viewModel: network)
        }
        .sheet(isPresented: $folderPickerShow) {
            FolderPicker { result in
                switch result {
                case .success(let folder):
                    folder.accessSecurityScopedResource { folder in
                        setAppFolder(url: folder)
                    }
                    mnist.load(from: getAppFolder()!)
                default: // .failure(let error)
                    break
                }
            }
        }
        // show performance in view
        // show statistics in view, e.g. durations, failed query images
        // save/ load trained model
        // train next x items
        // query random test item
        // train all items
        // query all test items
        // add pen query interface
        // use bnn, core ml, metal, (cuda)
        // use different datasets, e.g. cifar
        Button("train 100") {
            for i in 0..<100 {
                let input = (mnist.dataset[.images(.train)] as! [[UInt8]])[i]
                let target = (mnist.dataset[.labels(.train)] as! [UInt8])[i]
                network.train(for: input, with: target)
            }
        }
        Button("query") {
            let s = Int.random(in: 0..<10000)
            let input = (mnist.dataset[.images(.test)] as! [[UInt8]])[s]
            let target = (mnist.dataset[.labels(.test)] as! [UInt8])[s]
            let result = network.query(for: input)
            print("query \(target) yields \(result)")
        }
    }
}

extension MYONN {
    class MYONNViewModel: ObservableObject {
    }
}

// NYONN configuration for Network factory
let MYONNConfig: NetworkConfig = (
    layersWithSizes: [784, 100, 10], activationFunctions: [.sigmoid, .sigmoid], learningRate: 0.5
)

// specialized NYONN factory
struct MYONNFactory: AbstractFactory {
    func create(_ config: Never?) -> Network? {
        Network([
            Layer(numberOfInputs: 784, numberOfPUnits: 100, activationFunction: .sigmoid),
            Layer(numberOfInputs: 100, numberOfPUnits: 10, activationFunction: .sigmoid)
        ], alpha: 0.3)
    }
}
