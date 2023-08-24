import SwiftUI

struct MYONNView: View {
    @State private var folderPickerShow = getAppFolder() == nil
    
    @StateObject var viewModel = MYONNViewModel()
    
    @State private var falseResults = 0
    @State private var resultCorrect = true

    var body: some View {
        VStack {
            HStack {
                MNISTView(viewModel: viewModel.mnist)
                Spacer()
                Circle().foregroundColor(resultCorrect ? .red : .green)
            }
            ProgressView(value: viewModel.progressValue)
            HStack {
                Image(systemName: "figure.strengthtraining.traditional")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                Button {
                    Task {
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
                    resultCorrect = result == target
                    if !resultCorrect {
                        falseResults += 1
                    }
                } label: {
                    Image(systemName: "doc")
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
                Button {} label: {
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
        @Published var samplesQueried = 0
        
        @Published var progressValue: Float = 0
        
        func train(startWithSample index: Int, count: Int) async {
            samplesTrained += count
            progressValue = 0
            for i in index..<index + count {
                let input = (mnist.dataset[.images(.train)] as! [[UInt8]])[i]
                let target = (mnist.dataset[.labels(.train)] as! [UInt8])[i]
                await network.train(for: input, with: target)
                progressValue = Float(i) / Float(count)
            }
            progressValue = 0
        }
        
        func query(sample index: Int) -> (Int, Int) {
            samplesQueried += 1
            let input = (mnist.dataset[.images(.test)] as! [[UInt8]])[index]
            let target = (mnist.dataset[.labels(.test)] as! [UInt8])[index]
            let result = network.query(for: input).maxValueIndex()
            return (result, Int(target))
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
