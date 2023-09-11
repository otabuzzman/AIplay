import SwiftUI

struct MYONNView: View {
    @State private var folderPickerShow = getAppFolder() == nil
    
    @StateObject var viewModel = MYONNViewModel()
    
    @State private var queryResultCorrect: Bool?

    var body: some View {
        VStack {
            HStack {
                MNISTView(viewModel: viewModel.mnist)
                VStack {
                    Text("\(viewModel.performance)")
                    Text("\(viewModel.trainingDuration)")
                }
                QueryState(value: queryResultCorrect)
            }
            ProgressView(value: viewModel.trainingProgress)
            HStack {
                Image(systemName: "figure.strengthtraining.traditional")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                Button {
                    Task { @MainActor in
                        if viewModel.miniBatchSize == 1 {
                            // SGD with arbitrary number of samples
                            await viewModel.train(startWithSample: viewModel.samplesTrained, count: 100)
                        } else {
                            // mini-batch GD with size as configured
                            await viewModel.train(startWithBatch: viewModel.batchesTrained, count: 1)
                        }
                    }
                } label: {
                    Image(systemName: "doc.on.doc")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                Button {
                    Task { @MainActor in
                        await viewModel.trainAll()
                    }
                } label: {
                    Image(systemName: "book.closed")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
            }
            HStack {
                Image(systemName: "magnifyingglass")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                Button {
                    guard
                        let max = viewModel.mnist.dataset[.images(.test)]?.count
                    else { return }
                    var result: Int
                    var target: Int
                    (result, target) = viewModel.query(sample: Int.random(in: 0..<max))
                    queryResultCorrect = result == target
                } label: {
                    Image(systemName: "doc")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                Button {
                    Task { @MainActor in
                        await viewModel.queryAll()
                    }
                } label: {
                    Image(systemName: "book.closed")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
            }
            HStack {
                Button {
                    viewModel.reset()
                    queryResultCorrect = nil
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                NetworkView(viewModel: viewModel.network)
            }
        }
        .sheet(isPresented: $folderPickerShow) {
            FolderPicker { result in
                switch result {
                case .success(let folder):
                    folder.accessSecurityScopedResource { folder in
                        setAppFolder(url: folder)
                    }
                    viewModel.mnist.load(from: getAppFolder()!)
                default: // .failure(let error)
                    break
                }
            }
        }
    }
}

extension MYONNView {
    class MYONNViewModel: ObservableObject {
        var mnist = MNISTViewModel(in: getAppFolder())
        // specialized MYONN factory
        var network = NetworkViewModel(GenericFactory.create(MYONNFactory(), nil)!)
        // Network factory with MYONN configuration
        // var network = NetworkViewModel(GenericFactory.create(NetworkFactory(), MYONNConfig)!)
        
        @Published var samplesTrained = 0  
        private var samplesQueried = [Int]()
        @Published var batchesTrained = 0
        let miniBatchSize = 30
        @Published var epochsFinished = 0
        @Published var performance: Float = 0
        
        @Published var trainingProgress: Float = 0 // 0...1
        @Published var trainingDuration: TimeInterval = 0
        
        func trainAll() async -> Void {
            guard
                let count = mnist.dataset[.images(.train)]?.count
            else { return }
            if miniBatchSize == 1 {
                await train(startWithSample: 0, count: count)
            } else {
                await train(startWithBatch: 0, count: count / miniBatchSize)
            }
            epochsFinished += 1
        }
        
        func train(startWithSample index: Int, count: Int) async -> Void {
            trainingProgress = 0
            trainingDuration = 0
            let t0 = Date.timeIntervalSinceReferenceDate
            for i in 0..<count {
                let input = (mnist.dataset[.images(.train)] as! [[UInt8]])[index + i]
                let target = (mnist.dataset[.labels(.train)] as! [UInt8])[index + i]
                network.train(for: input, with: target)
                trainingProgress = Float(i + 1) / Float(count)
            }
            let t1 = Date.timeIntervalSinceReferenceDate
            trainingDuration = t1 - t0
            samplesTrained += count
            Task { @MainActor in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                trainingProgress = 0
            }
        }
        
        func train(startWithBatch index: Int, count: Int) async -> Void {
            trainingProgress = 0
            trainingDuration = 0
            let t0 = Date.timeIntervalSinceReferenceDate
            for i in 0..<count {
                let a = (i + index) * miniBatchSize
                let o = a + miniBatchSize
                let input = (mnist.dataset[.images(.train)] as! [[UInt8]])[a..<o]
                let target = (mnist.dataset[.labels(.train)] as! [UInt8])[a..<o]
                await network.train(for: input, with: target)
                trainingProgress = Float((i + 1)) / Float(count)
            }
            let t1 = Date.timeIntervalSinceReferenceDate
            trainingDuration = t1 - t0
            batchesTrained += count
            samplesTrained += count * miniBatchSize
            Task { @MainActor in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                trainingProgress = 0
            }
        }
        
        func queryAll() async -> Void {
            let sampleCount = mnist.dataset[.images(.test)]?.count ?? 0
            samplesQueried = [Int](repeating: .zero, count: sampleCount)
            await query(startWithSample: 0, count: sampleCount)
            performance = samplesQueried.count > 0 ? Float(samplesQueried.reduce(0, +)) / Float(samplesQueried.count) : 0
        }
        
        func query(startWithSample index: Int, count: Int) async -> Void {
            trainingProgress = 0
            for i in 0..<count {
                _ = query(sample: i)
                trainingProgress = Float(i) / Float(count - 1)
            }
            Task { @MainActor in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                trainingProgress = 0
            }
        }
        
        func query(sample index: Int) -> (Int, Int) {
            let input = (mnist.dataset[.images(.test)] as! [[UInt8]])[index]
            let target = (mnist.dataset[.labels(.test)] as! [UInt8])[index]
            let result = network.query(for: input).maxValueIndex()
            if samplesQueried.count > index {
                samplesQueried[index] = result == target ? 1 : 0
            }
            return (result, Int(target))
        }
        
        func reset() -> Void {
            network = NetworkViewModel(GenericFactory.create(MYONNFactory(), nil)!)
            samplesTrained = 0
            batchesTrained = 0
            epochsFinished = 0
            performance = 0
            trainingDuration = 0
        }
    }
}

// NYONN configuration for Network factory
let MYONNConfig: NetworkConfig = (
    layersWithSizes: [784, 100, 10], activationFunctions: [.sigmoid, .sigmoid], learningRate: 0.3
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

extension Matrix where Entry: Comparable {
    func maxValueEntry() -> Entry {
        entries.max(by: { $0 < $1 })! // probably save to force unwrap
    }
    
    func maxValueIndex() -> Int {
        entries.indices.max(by: { entries[$0] < entries[$1] })! // probably save to force unwrap
    }
}
