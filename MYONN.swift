import SwiftUI

struct MYONN: View {
    @State private var folderPickerShow = getAppFolder() == nil
    
    @StateObject var viewModel = MYONNViewModel()
    
    @StateObject var mnist = MNISTViewModel(in: getAppFolder())
    @StateObject var network = NetworkViewModel(
        layersWithSizes: [784, 100, 10],
        activationFunction: .sigmoid,
        learningRate: 0.3)
    
    var body: some View {
        HStack {
            MNIST(viewModel: mnist)
            Network(viewModel: network)
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

enum ActivationFunction {
    case identity
    case sigmoid
    
    var implementation: (Float) -> Float {
        switch self {
        case .identity:
            return { x in x }
        case .sigmoid:
            return { x in 1.0 / (1.0 + expf(-x)) }
        }
    }
}

extension NetworkViewModel {
    convenience init(
        layersWithSizes: [Int], activationFunction: ActivationFunction, learningRate: Float
    ) {
        var layers: [Layer] = []
        for i in 1..<layersWithSizes.count {
            let prevLayerSize = layersWithSizes[i - 1]
            let thisLayerSize = layersWithSizes[i]
            let layer = Layer(
                numberOfInputs: prevLayerSize,
                numberOfPUnits: thisLayerSize,
                activationFunction: activationFunction.implementation)
            layers.append(layer)
        }
        self.init(layers, alpha: learningRate)
    }
    
    func query(for I: [UInt8]) -> Matrix<Float> {
        let input = Matrix<Float>(rows: I.count, columns: 1, entries: I.map({ Float($0) }))
            .map { ($0 / 255.0 * 0.99) + 0.01 } // MYONN, p. 151 ff.
        return query(for: input)
    }
    
    func train(for I: [UInt8], with T: UInt8) -> Void {
        let input = Matrix<Float>(rows: I.count, columns: 1, entries: I.map({ Float($0) }))
            .map { ($0 / 255.0 * 0.99) + 0.01 }
        var target = Matrix<Float>(rows: 10, columns: 1)
            .map { _ in 0.01 }
        target[Int(T), 0] = 0.99
        train(for: input, with: target)
    }
}
