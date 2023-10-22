import SwiftUI
import PencilKit

import UniformTypeIdentifiers

struct NetworkView: View {
    @ObservedObject private var viewModel: NetworkViewModel

    @State private var longRunTask: Task<Void, Never>?
    @State private var longRunBusy = false
    
    @State private var canvas = PKCanvasView()
    @State private var canvasInput = MNISTImage()
    
    @State private var queryInput = MNISTImage()
    @State private var queryTarget: Int?
    @State private var resultReading: Int?
    @State private var resultDetails: [(Int, Float)]?
    
    @State private var resultDetailsShow = false
    
    @State private var error: NetworkExchangeError?
    
    @State private var isExporting = false
    @State private var isImporting = false
    
    var body: some View {
        ProgressView(value: viewModel.progress)
        VStack {
            HStack { // mimic list section header look
                Label("NN PREDICTION", systemImage: "wand.and.stars.inverse")
                    .foregroundColor(.secondary).bold(true)
                Spacer()
            }
            .padding(.leading, 12)
            .padding(.bottom, 2)
            HStack {
                Button {
                    let index = Int.random(in: 0..<viewModel.dataset.count(in: .test))
                    var target: MNISTLabel
                    (queryInput, target) = viewModel.dataset.fetch(index, from: .test)
                    queryTarget = Int(target)
                    let result = viewModel.query(sample: queryInput)
                    resultDetails = result.enumerated().sorted(by: { $0.element > $1.element })
                    resultReading = resultDetails?[0].0
                } label: {
                    Label("Random testset image", systemImage: "sparkle.magnifyingglass")
                }
                Spacer()
                Text("\(resultReading == nil ? "-" : resultReading!.description)")
                    .foregroundColor(resultReading == nil || resultReading == queryTarget ? .primary : .red)
                Button {
                    withAnimation {
                        resultDetailsShow.toggle()
                    }
                } label: {
                    Image(systemName: resultDetailsShow ? "chevron.down" : "chart.bar.doc.horizontal")
                }
                .frame(minWidth: 18)
                .disabled((resultDetails?.count ?? 0) == 0)
            }
            .padding()
            .background(.white)
            .background(in: RoundedRectangle(cornerRadius: 12))
            if resultDetailsShow {
                ResultDetailsView(resultDetails)
                    .padding(.leading)
            }
            HStack {
                VStack {
                    HStack {
                        Text("Query sketch")
                        Spacer()
                        Button {
                            canvas.drawing = PKDrawing()
                            resultReading = nil
                        } label: {
                            Image(systemName: "xmark.app")
                        }
                    }
                    ZStack {
                        Image(systemName: "square.and.pencil")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .padding()
                            .overlay(RoundedRectangle(cornerRadius: 4).stroke(.primary, lineWidth: 1))
                            .foregroundColor(Color(UIColor.secondarySystemBackground))
                        CanvasView(canvas: $canvas, mNISTImage: $canvasInput)
                            .aspectRatio(1, contentMode: .fit)
                            .onChange(of: canvasInput) { input in
                                queryInput = canvasInput
                                let result = viewModel.query(sample: queryInput)
                                resultDetails = result.enumerated().sorted(by: { $0.element > $1.element })
                                resultReading = resultDetails?[0].0
                                queryTarget = resultReading
                            }
                    }
                }
                .frame(minWidth: 0, maxWidth: .infinity)
                VStack {
                    Text("Query image")
                    ZStack {
                        Group {
                            Image(systemName: "sparkle.magnifyingglass")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .padding()
                            Image(mNISTImage: queryInput)?
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .colorInvert()
                        }
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(.primary, lineWidth: 1))
                        .foregroundColor(Color(UIColor.secondarySystemBackground))
                    }
                }
                .frame(minWidth: 0, maxWidth: .infinity)
            }
            .padding()
            .background(.white)
            .background(in: RoundedRectangle(cornerRadius: 12))
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        VStack {
            Form {
                Section {
                    // https://rhonabwy.com/2021/02/13/nested-observable-objects-in-swiftui/
                    MNISTDatasetView(viewModel: viewModel.dataset, busy: $longRunBusy)
                        .disabled(longRunBusy)
                } header: {
                    HStack {
                        Label("DATASET", systemImage: "chart.bar").font(.headline)
                        Spacer()
                    }
                }
                Section {
                    HStack {
                        Text("Mini-batch size")
                        Spacer()
                        Text("\(viewModel.miniBatchSize)")
                    }
                    HStack {
                        Text("Learning rate")
                        Spacer()
                        Text("0.3")
                    }
                } header: {
                    HStack {
                        Label("NN CONFIGURATION", systemImage: "gearshape").font(.headline)
                        Spacer()
                    }
                }
                Section {
                    HStack {
                        Text("Epochs trained so far")
                        Spacer()
                        Text("\(viewModel.epochsTrained)")
                    }
                    HStack {
                        Text("Duration of last epoch")
                        Spacer()
                        Text(DateComponentsFormatter().string(from: viewModel.duration)!)
                    }
                } header: {
                    HStack {
                        Label("NN TRAINING", systemImage: "dumbbell").font(.headline)
                        Spacer()
                    }
                }
                Section {
                    Group {
                        HStack {
                            Button {
                                longRunTask = Task { @MainActor in
                                    longRunBusy = true
                                    if viewModel.miniBatchSize == 1 {
                                        // SGD with arbitrary number of samples
                                        await viewModel.train(startWithSample: viewModel.samplesTrained, count: 100)
                                    } else {
                                        // mini-batch GD with size as configured
                                        await viewModel.train(startWithBatch: viewModel.batchesTrained, count: 1)
                                    }
                                    longRunBusy = false
                                }
                            } label: {
                                Label("Train next mini-batch", systemImage: "figure.strengthtraining.traditional")
                            }
                            Spacer()
                        } 
                        HStack {
                            Button {
                                longRunTask = Task { @MainActor in
                                    longRunBusy = true
                                    await viewModel.trainAll()
                                    longRunBusy = false
                                }
                            } label: {
                                Label("Train another epoch", systemImage: "figure.strengthtraining.traditional")
                            }
                            Spacer()
                        }
                        HStack {
                            Button {
                                longRunTask = Task { @MainActor in
                                    longRunBusy = true
                                    await viewModel.queryAll()
                                    longRunBusy = false
                                }
                            } label: {
                                Label("NN Accuracy", systemImage: "sparkle.magnifyingglass")
                            }
                            Spacer()
                            Text(String(format: "%.4f", viewModel.accuracy))
                        }
                        HStack {
                            Button {
                                isImporting = true
                            } label: {
                                Label("Import model from Files", systemImage: "square.and.arrow.up")
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
                            Spacer()
                        }
                        HStack {
                            Button {
                                isExporting = true
                            } label: {
                                Label("Export model to Files", systemImage: "square.and.arrow.down")
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
                            Spacer()
                        }
                    }
                    .disabled(longRunBusy)
                } footer: {
                    HStack {
                        Spacer()
                        Button("Reset", role: .destructive) {
                            Task { @MainActor in
                                longRunTask?.cancel()
                                let _ = await longRunTask?.value
                                viewModel.reset()
                                queryInput = []
                                queryTarget = nil
                                withAnimation() {
                                    resultDetailsShow = false
                                }
                                resultDetails = nil
                                resultReading = nil
                            }
                        }
                        Spacer()
                    }
                    .padding()
                }
            }
        }
    }
}

extension NetworkView {
    init?(config: NetworkConfig) {
        guard
            let network = GenericFactory.create(NetworkFactory(), config)
        else { return nil }
        viewModel = NetworkViewModel(network, MNISTDatasetViewModel())
    }
}

class NetworkViewModel: ObservableObject {
    var network: Network
    var dataset: MNISTDatasetViewModel
    
    private(set) var miniBatchSize = 30
    
    @Published private(set) var samplesTrained = 0  
    private var samplesQueried = [Int]()
    @Published private(set) var batchesTrained = 0
    
    @Published private(set) var epochsTrained = 0
    @Published private(set) var accuracy: Float = 0
    
    @Published private(set) var progress: Float = 0 // 0...1
    private let progressIncrement: Float = 0.01 // 0>..1
    
    @Published private(set) var duration: TimeInterval = 0
    
    init(_ network: Network, _ dataset: MNISTDatasetViewModel) {
        self.network = network
        self.dataset = dataset
    }
    
    func queryAll() async -> Void {
        let sampleCount = dataset.count(in: .test)
        samplesQueried = [Int](repeating: .zero, count: sampleCount)
        await query(startWithSample: 0, count: sampleCount)
        accuracy = samplesQueried.count > 0 ? Float(samplesQueried.reduce(0, +)) / Float(samplesQueried.count) : 0
    }
    
    private func query(startWithSample index: Int, count: Int) async -> Void {
        for i in 0..<count {
            let (input, target) = dataset.fetch(i, from: .test)
            let result = query(sample: input).maxElementIndex()! // probably save to unwrap
            if samplesQueried.count > index {
                samplesQueried[i] = result == target ? 1 : 0
            }
            let progress = Float(i + 1) / Float(count)
            if progress > self.progress + progressIncrement {
                self.progress = progress
            }
            do {
                try Task.checkCancellation()
            } catch { return }
        }
        Task { @MainActor in
            try await Task.sleep(nanoseconds: 1_000_000_000)
            progress = 0
        }
    }
    
    func query(sample: MNISTImage) -> [Float] {
        let I = Matrix<Float>(
            rows: sample.count, columns: 1,
            entries: sample.map { (Float($0) / 255.0 * 0.99) + 0.01 }) // MYONN, p. 151 ff.
        return network.query(for: I).entries
    }
    
    func trainAll() async -> Void {
        dataset.shuffle()
        let count = dataset.count(in: .train)
        if miniBatchSize == 1 {
            await train(startWithSample: 0, count: count)
        } else {
            await train(startWithBatch: 0, count: count / miniBatchSize)
        }
        epochsTrained += 1
    }
    
    func train(startWithSample index: Int, count: Int) async -> Void {
        duration = 0
        let t0 = Date.timeIntervalSinceReferenceDate
        for i in 0..<count {
            let (input, target) = dataset.fetch(index + 1, from: .train)
            train(sample: input, target: target)
            let progress = Float(i + 1) / Float(count)
            if progress > self.progress + progressIncrement {
                self.progress = progress
            }
            do {
                try Task.checkCancellation()
            } catch { return }
        }
        let t1 = Date.timeIntervalSinceReferenceDate
        duration = t1 - t0
        samplesTrained += count
        Task { @MainActor in
            try await Task.sleep(nanoseconds: 1_000_000_000)
            progress = 0
        }
    }
    
    func train(sample: MNISTImage, target: MNISTLabel) -> Void {
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
            let (input, target) = dataset.fetch(a..<o, from: .train)
            let I = input.map {
                Matrix<Float>(
                    rows: $0.count, columns: 1,
                    entries: $0.map { (Float($0) / 255.0 * 0.99) + 0.01 })
            }
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
            do {
                try Task.checkCancellation()
            } catch { return }
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
        epochsTrained = 0
        accuracy = 0
        progress = 0
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

enum NetworkExchangeError: Error {
    case nndxSave(Error)
    case nndxLoad(Error)
    case nndxRead(URL, Error)
    case nndxDecode(URL)
}

extension NetworkExchangeError {
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

extension Array where Element: Comparable {
    func maxElementValue() -> Element? {
        self.max(by: { $0 < $1 })
    }
    
    func maxElementIndex() -> Int? {
        self.indices.max(by: { self[$0] < self[$1] })
    }
}

struct ResultDetailsView: View {
    let details: [(Int, Float)]
    
    init?(_ details: [(Int, Float)]?) {
        guard
            let details = details
        else { return nil }
        self.details = details
    }
    
    var body: some View {
        VStack(spacing: 2) {
            ForEach(0..<details.count, id: \.self) { index in
                HStack {
                    Text("\(details[index].0)")
                        .frame(minWidth: 12)
                    ProgressView(value: details[index].1)
                    Text("\(details[index].1)")
                        .frame(minWidth: 96)
                }
            }
        }
    }
}
