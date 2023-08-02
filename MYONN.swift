import SwiftUI

struct MYONN: View {
    @StateObject var mnist = MNIST(in: getAppFolder())
    @StateObject var network = Network(
        layersWithSizes: [784, 100, 10],
        activationFunction: { x in 1.0 / (1.0 + expf(-x)) }, learningRate: 0.3)
    
    @State private var folderPickerShow = getAppFolder() == nil
    
    var body: some View {
        HStack {
            VStack {
                Circle().fill(mnist.dataset[.images(.train)] == nil ? .gray : .green)
                Circle().fill(mnist.dataset[.labels(.train)] == nil ? .gray : .green)
            }
            VStack {
                Circle().fill(mnist.dataset[.images(.test)] == nil ? .gray : .green)
                Circle().fill(mnist.dataset[.labels(.test)] == nil ? .gray : .green)
            }
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

extension Network {
    convenience init(
        layersWithSizes: [Int], activationFunction: @escaping (Float) -> Float, learningRate: Float
    ) {
        var layers: [Layer] = []
        for i in 1..<layersWithSizes.count {
            let prevLayerSize = layersWithSizes[i - 1]
            let thisLayerSize = layersWithSizes[i]
            let layer = Layer(
                numberOfInputs: prevLayerSize,
                numberOfPUnits: thisLayerSize,
                activationFunction: activationFunction)
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
