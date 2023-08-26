import SwiftUI

struct MYONNView: View {
    @State private var folderPickerShow = getAppFolder() == nil
    
    @StateObject var viewModel = MYONNViewModel()
    
    @State private var queryResultCorrect: Bool?

    var body: some View {
        VStack {
            HStack {
                MNISTView(viewModel: viewModel.mnist)
                Spacer()
                QueryState(value: queryResultCorrect)
            }
            ProgressView(value: viewModel.progressValue)
            HStack {
                Image(systemName: "figure.strengthtraining.traditional")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                Button {
                    Task { @MainActor in
                        await viewModel.train(startWithSample: viewModel.samplesTrained, count: 100)
                    }
                } label: {
                    Image(systemName: "doc.on.doc")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                Button {} label: {
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
                    var result: Int
                    var target: Int
                    (result, target) = viewModel.query(sample: Int.random(in: 0..<10000))
                    queryResultCorrect = result == target
                } label: {
                    Image(systemName: "doc")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                Button {
                    Task { @MainActor in
                        await viewModel.query(startWithSample: 0, count: 1000)
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
        @Published var samplesQueried = [Int]()
        
        @Published var progressValue: Float = 0 // 0...1
        
        func train(startWithSample index: Int, count: Int) async -> Void {
            progressValue = 0
            for i in 0..<count {
                let input = (mnist.dataset[.images(.train)] as! [[UInt8]])[index + i]
                let target = (mnist.dataset[.labels(.train)] as! [UInt8])[index + i]
                network.train(for: input, with: target)
                progressValue = Float(i) / Float(count - 1)
            }
            samplesTrained += count
            Task { @MainActor in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                progressValue = 0
            }
        }
        
        func query(startWithSample index: Int, count: Int) async -> Void {
            progressValue = 0
            for i in 0..<count {
                _ = query(sample: i)
                progressValue = Float(i) / Float(count - 1)
            }
            Task { @MainActor in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                progressValue = 0
            }
        }
        
        func query(sample index: Int) -> (Int, Int) {
            let input = (mnist.dataset[.images(.test)] as! [[UInt8]])[index]
            let target = (mnist.dataset[.labels(.test)] as! [UInt8])[index]
            let result = network.query(for: input).maxValueIndex()
            samplesQueried.append(result == target ? 1 : 0)
            return (result, Int(target))
        }
        
        func reset() -> Void {
            network = NetworkViewModel(GenericFactory.create(MYONNFactory(), nil)!)
            samplesTrained = 0
            samplesQueried = []
        }
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

extension Matrix where Entry: Comparable {
    func maxValueEntry() -> Entry {
        entries.max(by: { $0 < $1 })! // probably save to force unwrap
    }
    
    func maxValueIndex() -> Int {
        entries.indices.max(by: { entries[$0] < entries[$1] })! // probably save to force unwrap
    }
}

struct QueryState: View {
    var value: Bool?
    
    var body: some View {
        Circle().foregroundColor(value == nil ? .gray : value! ? .green : .red)
    }
}
