import SwiftUI

struct MYONN: View {
    @StateObject var mnist = MNIST()
    @StateObject var network = Network([
        Layer(numberOfInputs: 784, numberOfPUnits: 100, activationFunction: { x in 1.0 / (1.0 + expf(-x)) }),
        Layer(numberOfInputs: 100, numberOfPUnits: 10, activationFunction: { x in 1.0 / (1.0 + expf(-x)) })
    ], learningRate: 0.3)
    
    @State private var folderPickerShow = getAppFolder() == nil
    
    var body: some View {
        Circle().foregroundColor(
            mnist.dataset.count == 0 ? .gray :
                mnist.dataset.count == 4 ? .green :
                    .yellow
        )
        .task {
            mnist.load(from: getAppFolder())
        }
        .sheet (isPresented: $folderPickerShow) {
            FolderPicker { result in
                switch result {
                case .success(let folder):
                    folder.accessSecurityScopedResource { folder in
                        setAppFolder(url: folder)
                    }
                    mnist.load(from: getAppFolder())
                default: // .failure(let error)
                    break
                }
            }
        }
        Text("\(mnist.dataset[.images(.train)] == nil ? 0 : (mnist.dataset[.images(.train)] as! [[UInt8]]).count)")
        Text("\(mnist.dataset[.images(.test)] == nil ? 0 : (mnist.dataset[.images(.test)] as! [[UInt8]]).count)")
        Text("\(mnist.dataset[.labels(.train)] == nil ? 0 : (mnist.dataset[.labels(.train)] as! [UInt8]).count)")
        Text("\(mnist.dataset[.labels(.test)] == nil ? 0 : (mnist.dataset[.labels(.test)] as! [UInt8]).count)")
        Button("probe") {
            print((mnist.dataset[.images(.test)] as! [[UInt8]])[123])
            print((mnist.dataset[.labels(.test)] as! [UInt8])[123])
        }
    }
}

extension Network {
    func query(for I: [UInt8]) -> Matrix<Float> {
        let i = Matrix<Float>(rows: I.count, columns: 1, entries: I.map({ Float($0) }))
            .map { ($0 / 255.0 * 0.99) + 0.01 } // MYONN, p. 151 ff.
        return query(for: i)
    }
    
    func train(for I: [UInt8], with T: UInt8) -> Void {
        let i = Matrix<Float>(rows: I.count, columns: 1, entries: I.map({ Float($0) }))
            .map { ($0 / 255.0 * 0.99) + 0.01 }
        var t = Matrix<Float>(rows: 10, columns: 1)
            .map { _ in 0.01 }
        t[Int(T) - 1, 0] = 0.99
        return train(for: i, with: t)
    }
}
