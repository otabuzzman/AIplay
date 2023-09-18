import SwiftUI
import UniformTypeIdentifiers

struct NetworkView: View {
    @ObservedObject private var viewModel: NetworkViewModel

    @State private var queryResultCorrect: Bool?
    
    var body: some View {
        HStack {
            VStack {
                Circle().foregroundColor(viewModel.dataset.state[.images(.train)]?.color)
                Circle().foregroundColor(viewModel.dataset.state[.labels(.train)]?.color)
            }
            VStack {
                Circle().foregroundColor(viewModel.dataset.state[.images(.test)]?.color)
                Circle().foregroundColor(viewModel.dataset.state[.labels(.test)]?.color)
            }
        }
        ProgressView(value: viewModel.progress)
        HStack {
            Button {
                viewModel.reset()
                queryResultCorrect = nil
            } label: {
                Image(systemName: "arrow.counterclockwise")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            VStack {
                Text("\(viewModel.performance)")
                Text("\(viewModel.duration)")
            }
            Circle().foregroundColor(queryResultCorrect == nil ? .gray : queryResultCorrect! ? .green : .red)
        }
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
            Image(systemName: "sparkle.magnifyingglass")
                .resizable()
                .aspectRatio(contentMode: .fit)
            Button {
                guard
                    let max = viewModel.dataset.subsets[.images(.test)]?.count
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
        NetworkExchangeView(viewModel: viewModel)
        QueryView(viewModel: viewModel)
    }
}

// https://swiftunwrap.com/article/handling-optional-in-swiftui-view
extension NetworkView {
    init?(dataset: MNISTDataset) {
        guard
            let network = GenericFactory.create(NetworkFactory(), defaultConfig)
        else { return nil }
        viewModel = NetworkViewModel(network, dataset)
    }
}

class NetworkViewModel: ObservableObject {
    var network: Network
    var dataset: MNISTDataset
    
    private(set) var miniBatchSize = 30
    
    @Published var samplesTrained = 0  
    private var samplesQueried = [Int]()
    @Published var batchesTrained = 0
    
    @Published var epochsFinished = 0
    @Published var performance: Float = 0
    
    @Published var progress: Float = 0 // 0...1
    private let progressIncrement: Float = 0.01 // 0>..1
    
    @Published var duration: TimeInterval = 0
    
    init(_ network: Network, _ dataset:  MNISTDataset) {
        self.network = network
        self.dataset = dataset
    }
    
    func queryAll() async -> Void {
        let sampleCount = dataset.subsets[.images(.test)]?.count ?? 0
        samplesQueried = [Int](repeating: .zero, count: sampleCount)
        await query(startWithSample: 0, count: sampleCount)
        performance = samplesQueried.count > 0 ? Float(samplesQueried.reduce(0, +)) / Float(samplesQueried.count) : 0
    }
    
    func query(startWithSample index: Int, count: Int) async -> Void {
        for i in 0..<count {
            _ = query(sample: i)
            let progress = Float(i + 1) / Float(count)
            if progress > self.progress + progressIncrement {
                self.progress = progress
            }
        }
        Task { @MainActor in
            try await Task.sleep(nanoseconds: 1_000_000_000)
            progress = 0
        }
    }
    
    func query(sample index: Int) -> (Int, Int) {
        let input = (dataset.subsets[.images(.test)] as! [[UInt8]])[index]
        let target = (dataset.subsets[.labels(.test)] as! [UInt8])[index]
        let result = query(sample: input)
        if samplesQueried.count > index {
            samplesQueried[index] = result == target ? 1 : 0
        }
        return (result, Int(target))
    }
    
    func query(sample: [UInt8]) -> Int {
        let I = Matrix<Float>(
            rows: sample.count, columns: 1,
            entries: sample.map { (Float($0) / 255.0 * 0.99) + 0.01 }) // MYONN, p. 151 ff.
        return network.query(for: I).maxValueIndex()
    }
    
    func trainAll() async -> Void {
        guard
            let count = dataset.subsets[.images(.train)]?.count
        else { return }
        if miniBatchSize == 1 {
            await train(startWithSample: 0, count: count)
        } else {
            await train(startWithBatch: 0, count: count / miniBatchSize)
        }
        epochsFinished += 1
    }
    
    func train(startWithSample index: Int, count: Int) async -> Void {
        duration = 0
        let t0 = Date.timeIntervalSinceReferenceDate
        for i in 0..<count {
            let input = (dataset.subsets[.images(.train)] as! [[UInt8]])[index + i]
            let target = (dataset.subsets[.labels(.train)] as! [UInt8])[index + i]
            train(sample: input, target: target)
            let progress = Float(i + 1) / Float(count)
            if progress > self.progress + progressIncrement {
                self.progress = progress
            }
        }
        let t1 = Date.timeIntervalSinceReferenceDate
        duration = t1 - t0
        samplesTrained += count
        Task { @MainActor in
            try await Task.sleep(nanoseconds: 1_000_000_000)
            progress = 0
        }
    }
    
    func train(sample: [UInt8], target: UInt8) -> Void {
        let I = Matrix<Float>(
            rows: sample.count, columns: 1,
            entries: sample.map { (Float($0) / 255.0 * 0.99) + 0.01 })
        var T = Matrix<Float>(rows: 10, columns: 1)
            .map { _ in 0.01 }
        T[Int(target), 0] = 0.99
        network.train(for: I, with: T)
    }
    
    func train(startWithBatch index: Int, count: Int) async -> Void {
        duration = 0
        let t0 = Date.timeIntervalSinceReferenceDate
        for i in 0..<count {
            let a = (index + i) * miniBatchSize
            let o = a + miniBatchSize
            let input = (dataset.subsets[.images(.train)] as! [[UInt8]])[a..<o]
            let I = input.map {
                Matrix<Float>(
                    rows: $0.count, columns: 1,
                    entries: $0.map { (Float($0) / 255.0 * 0.99) + 0.01 })
            }
            let target = (dataset.subsets[.labels(.train)] as! [UInt8])[a..<o]
            let T = target.map {
                var target = Matrix<Float>(rows: 10, columns: 1)
                    .map { _ in 0.01 }
                target[Int($0), 0] = 0.99
                return target
            }
            await network.train(for: I, with: T)
            let progress = Float(i + 1) / Float(count)
            if progress > self.progress + progressIncrement {
                self.progress = progress
            }
        }
        let t1 = Date.timeIntervalSinceReferenceDate
        duration = t1 - t0
        batchesTrained += count
        samplesTrained += count * miniBatchSize
        Task { @MainActor in
            try await Task.sleep(nanoseconds: 1_000_000_000)
            progress = 0
        }
    }
    
    func reset() -> Void {
        network = GenericFactory.create(NetworkFactory(), defaultConfig)!
        samplesTrained = 0
        batchesTrained = 0
        epochsFinished = 0
        performance = 0
        duration = 0
    }
}

enum NetworkExchangeDocumentError: Error {
    case modified
}

extension NetworkExchangeDocumentError {
    var description: String {
        switch self {
        case .modified:
            return "document modified while reading"
        }
    }
}

extension UTType {
    public static let nnxd = UTType(exportedAs: "com.otabuzzman.aiplay.nnxd")
}

struct NetworkExchangeDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.nnxd] }
    static var writableContentTypes: [UTType] { [.nnxd] }
    
    private(set) var content: Data
    
    init(_ content: Data) {
        self.content = content
    }
    
    init(configuration: FileDocumentReadConfiguration) throws {
        guard
            let content = configuration.file.regularFileContents
        else { throw NetworkExchangeDocumentError.modified }
        self.content = content
    }
    
    func fileWrapper(configuration: FileDocumentWriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: content)
    }
}

enum NetworkExchangeViewError: Error {
    case nndxSave(Error)
    case nndxLoad(Error)
    case nndxRead(URL, Error)
    case nndxDecode(URL)
}

extension NetworkExchangeViewError {
    var description: String {
        switch self {
        case .nndxSave(let error):
            return "save NNDX file failed with error \(error)"
        case .nndxLoad(let error):
            return "load NNDX file failed with error \(error)"
        case .nndxRead(let url, let error):
            return "read NNDX file \(url) failed with error \(error)"
        case .nndxDecode(let url):
            return "decode NNDX file \(url) failed"
        }
    }
}

struct NetworkExchangeView: View {
    var viewModel: NetworkViewModel
    
    @State private var error: NetworkExchangeViewError? = nil
    
    @State private var isExporting = false
    @State private var isImporting = false
    
    var body: some View {
        HStack {
            Image(systemName: "externaldrive")
                .resizable()
                .aspectRatio(contentMode: .fit)
            Button {
                isExporting = true
            } label: {
                Image(systemName: "square.and.arrow.down")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            .fileExporter(isPresented: $isExporting,
                          document: NetworkExchangeDocument(viewModel.network.encode),
                          contentType: .nnxd, defaultFilename: "Untitled") { result in
                switch result {
                case .success:
                    break
                case .failure(let error):
                    self.error = .nndxSave(error)
                }
            }
            Button {
                isImporting = true
            } label: {
                Image(systemName: "square.and.arrow.up")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            .fileImporter(isPresented: $isImporting,
                          allowedContentTypes: [.nnxd], allowsMultipleSelection: false) { result in
                switch result {
                case .success(let url):
                    do {
                        let content = try Data(contentsOf: url[0])
                        guard
                            let network = Network(from: content)
                        else {
                            self.error = .nndxDecode(url[0])
                            return
                        }
                        viewModel.network = network
                    } catch {
                        self.error = .nndxRead(url[0], error)
                    }
                case .failure(let error):
                    self.error = .nndxLoad(error)
                }
            }
        }
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
