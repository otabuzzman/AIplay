import SwiftUI
import PencilKit

import UniformTypeIdentifiers

struct NetworkView: View {
    @ObservedObject private var viewModel = NetworkViewModel()
    @State private var actConfig = getNetworkConfig() ?? .default
    
    @State private var longRunTask: Task<Void, Never>?
    @State private var longRunBusy = false
    
    @State private var datasetReady = false
    @State private var datasetError = false
    
    @State private var canvas = PKCanvasView()
    @State private var canvasInput = MNISTImage()
    
    @State private var queryInput = MNISTImage()
    @State private var queryTarget: Int?
    @State private var resultReading: Int?
    @State private var resultDetails: [(Int, Float)]?
    
    @State private var showResultDetails = false
    
    @State private(set) var samplesTrained = 0
    
    @State private var validationAccuracy: Float?
    
    @State private var document: NetworkExchangeDocument?
    @State private var error: NetworkExchangeError?
    
    @State private var isExporting = false
    @State private var isImporting = false

    @State private var showSetupView = false
    
    var body: some View {
        ProgressView(value: viewModel.progress)
        LazyVStack {
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
                .disabled(longRunBusy || !datasetReady)
                Spacer()
                Text("\(resultReading == nil ? "-" : resultReading!.description)")
                    .foregroundColor(resultReading == nil || resultReading == queryTarget ? .primary : .red)
                Button {
                    withAnimation {
                        showResultDetails.toggle()
                    }
                } label: {
                    Image(systemName: showResultDetails ? "chevron.down" : "chart.bar.doc.horizontal")
                }
                .font(.title3)
                .disabled((resultDetails?.count ?? 0) == 0)
            }
            .padding()
            .background(Color(UIColor.systemBackground))
            .background(in: RoundedRectangle(cornerRadius: 12))
            if showResultDetails {
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
                        .font(.title3)
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
                            MSwitch {
                                Image(mNISTImage: queryInput)?
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .colorInvert()
                            } dark: {
                                Image(mNISTImage: queryInput)?
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                            }
                        }
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(.primary, lineWidth: 1))
                        .foregroundColor(Color(UIColor.secondarySystemBackground))
                    }
                }
                .frame(minWidth: 0, maxWidth: .infinity)
            }
            .padding()
            .background(Color(UIColor.systemBackground))
            .background(in: RoundedRectangle(cornerRadius: 12))
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        VStack {
            Form {
                Section {
                    // https://rhonabwy.com/2021/02/13/nested-observable-objects-in-swiftui/
                    MNISTView(viewModel: viewModel.dataset, ready: $datasetReady, error: $datasetError)
                        .disabled(!datasetReady && !datasetError || longRunBusy)
                } header: {
                    HStack {
                        Label("DATASET", systemImage: "chart.bar").font(.headline)
                        Spacer()
                    }
                }
                Section {
                    HStack {
                        Text("Statistics...")
                        Spacer()
                    }
                } header: {
                    HStack {
                        Label("NN TRAINING", systemImage: "dumbbell").font(.headline)
                        Spacer()
                        Button {
                            showSetupView.toggle()
                        } label: {
                            Image(systemName: "gearshape")
                        }
                        .font(.title3)
                    }
                }
                Section {
                    Group {
                        HStack {
                            Button {
                                longRunTask = Task { @MainActor in
                                    longRunBusy = true
                                    if viewModel.miniBatchSize == 1 {
                                        // SGD with mini-batch number of samples
                                        await viewModel.train(startWithSample: samplesTrained, count: viewModel.miniBatchSize)
                                    } else {
                                        // mini-batch GD with size as configured
                                        let batchesTrained = samplesTrained / viewModel.miniBatchSize
                                        await viewModel.train(startWithBatch: batchesTrained, count: 1)
                                    }
                                    samplesTrained += viewModel.miniBatchSize
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
                                    validationAccuracy = await viewModel.queryAll()
                                    longRunBusy = false
                                }
                            } label: {
                                Label("Validation Accuracy", systemImage: "sparkle.magnifyingglass")
                            }
                            Spacer()
                            Text("\(validationAccuracy == nil ? "-" : String(format: "%.4f", validationAccuracy!))")
                        }
                        HStack {
                            Button {
                                isImporting = true
                            } label: {
                                Label("Import model from Files", systemImage: "square.and.arrow.up")
                            }
                            Spacer()
                        }
                        HStack {
                            Button {
                                document = NetworkExchangeDocument(viewModel.network.encode)
                                isExporting = true
                            } label: {
                                Label("Export model to Files", systemImage: "square.and.arrow.down")
                            }
                            Spacer()
                        }
                    }
                    .disabled(longRunBusy || !datasetReady)
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
                                    showResultDetails = false
                                }
                                resultDetails = nil
                                resultReading = nil
                                samplesTrained = 0
                                validationAccuracy = nil
                            }
                        }
                        Spacer()
                    }
                    .padding()
                }
            }
        }
        .fileImporter(isPresented: $isImporting, allowedContentTypes: [.nnxd], allowsMultipleSelection: false) { result in
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
        // modifier attached to Button reopens system interface on pressing return and hide keys
        .fileExporter(isPresented: $isExporting, document: document, contentType: .nnxd, defaultFilename: "Untitled") { result in
            switch result {
            case .success:
                break
            case .failure(let error):
                self.error = .nndxSave(error)
            }
        }
        .sheet(isPresented: $showSetupView) {
            NetworkSetupView(isPresented: $showSetupView, actConfig) { newConfig in
                guard
                    let network = GenericFactory.create(NetworkFactory(), newConfig)
                else { return }
                viewModel.network = network
                viewModel.miniBatchSize = newConfig.miniBatchSize
                
                actConfig = newConfig
                setNetworkConfig(newConfig)
                
                showSetupView.toggle()
                
                if _isDebugAssertConfiguration() {
                    let _ = print(newConfig)
                }
            }
        }
    }
}

extension NetworkView {
    class NetworkViewModel: ObservableObject {
        var network: Network
        private(set) var dataset: MNISTViewModel
        private(set) var epochsWanted: Int
        var miniBatchSize: Int
        
        @Published private(set) var progress: Float = 0 // 0...1
        private let progressIncrement: Float = 0.01 // 0>..1
        
        init(config: NetworkConfig = .default) {
            if let model = Bundle.main.url(forResource: "default-model", withExtension: "nndx") {
                network = try! Network(from: Data(contentsOf: model))! // should not fail
            } else {
                network = GenericFactory.create(NetworkFactory(), config)! // probably save to unwrap
            }
            dataset = MNISTViewModel()
            epochsWanted = config.epochsWanted
            miniBatchSize = config.miniBatchSize
        }
        
        func queryAll(use subset: MNISTSubset.Purpose = .test) async -> Float {
            let sampleCount = dataset.count(in: subset)
            var samplesQueried = [Int](repeating: .zero, count: sampleCount)
            for index in 0..<sampleCount {
                let (input, target) = dataset.fetch(index, from: subset)
                let result = query(sample: input).maxElementIndex()! // probably save to unwrap
                samplesQueried[index] = result == target ? 1 : 0
                let progress = Float(index + 1) / Float(sampleCount)
                if progress > self.progress + progressIncrement {
                    self.progress = progress
                }
                do {
                    try Task.checkCancellation()
                } catch { return 0 }
            }
            Task { @MainActor in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                progress = 0
            }
            return samplesQueried.count > 0 ? Float(samplesQueried.reduce(0, +)) / Float(samplesQueried.count) : 0
        }
        
        func query(sample: MNISTImage) -> [Float] {
            network.query(for: sample.toInput()).entries
        }
        
        func trainAll() async -> Void {
            let measures = Measures()
            measures.trainingStartTime = Date.timeIntervalSinceReferenceDate
            dataset.shuffle()
            let count = dataset.count(in: .train)
            if miniBatchSize == 1 {
                await train(startWithSample: 0, count: count, measures)
            } else {
                await train(startWithBatch: 0, count: count / miniBatchSize, measures)
            }
            measures.trainingAccuracy = await queryAll(use: .train)
            measures.validationAccuracy = await queryAll(use: .test)
            measures.trainingDuration = Date.timeIntervalSinceReferenceDate - measures.trainingStartTime
            
            if _isDebugAssertConfiguration() {
                let _ = print(measures)
            }
        }
        
        func train(startWithSample index: Int, count: Int, _ measures: Measures? = nil) async -> Void {
            if let measures = measures, count > miniBatchSize {
                measures.trainingLoss = .init(repeating: 0, count: count / miniBatchSize)
            }
            for i in 0..<count {
                let (input, target) = dataset.fetch(index + i, from: .train)
                let trainingLoss = network.train(for: input.toInput(), with: target.toTarget())
                if i % miniBatchSize == 0 {
                    measures?.trainingLoss?[i] = trainingLoss
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
        
        func train(startWithBatch index: Int, count: Int, _ measures: Measures? = nil) async -> Void {
            if let measures = measures {
                measures.trainingLoss = .init(repeating: 0, count: count)
            }
            for i in 0..<count {
                let a = (index + i) * miniBatchSize
                let o = a + miniBatchSize
                let (input, target) = dataset.fetch(a..<o, from: .train)
                let trainingCost = await network.train(for: input.toMatrix(), with: target.toMatrix())
                measures?.trainingLoss?[i] = trainingCost
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
        
        func reset() -> Void {
            network = GenericFactory.create(NetworkFactory(), .default)!
            progress = 0
        }
    }
}

extension Array {
    func toMatrix() -> [Matrix<Float>] where Element == MNISTImage {
        self.map { $0.toInput() }
    }
    
    func toMatrix() -> [Matrix<Float>] where Element == MNISTLabel {
        self.map { $0.toTarget() }
    }
}

extension MNISTImage {
    func toInput() -> Matrix<Float> {
        Matrix<Float>(
            rows: self.count, columns: 1,
            entries: self.map { (Float($0) / 255.0 * 0.99) + 0.01 }) // MYONN, p. 151 ff.
    }
}

extension MNISTLabel {
    func toTarget() -> Matrix<Float> {
        var target = Matrix<Float>(rows: 10, columns: 1)
            .map { _ in 0.01 }
        target[Int(self), 0] = 0.99
        return target
    }
}

class Measures {
    var trainingStartTime: TimeInterval = 0
    var trainingDuration: TimeInterval = 0
    var trainingAccuracy: Float = 0
    var validationAccuracy: Float = 0
    var trainingLoss: [Float]?
}

extension Measures: CustomStringConvertible {
    var description: String {
        "Measures(trainingStartTime: \(trainingStartTime), trainingDuration: \(trainingDuration), trainingAccuracy: \(trainingAccuracy), validationAccuracy: \(validationAccuracy), trainingLoss: \(trainingLoss == nil ? "nil" : "[\(stringOfElements(in: trainingLoss!, count: 16, format: { String(describing: $0) }))])")"
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
                    ProgressView(value: details[index].1)
                    Text("\(details[index].1)")
                }
                .monospaced(true)
            }
        }
    }
}

struct MSwitch<T: View, U: View>: View {
    @Environment(\.colorScheme) var colorScheme
    
    let light: T
    let dark: U
    
    init(light: T, dark: U) {
        self.light = light
        self.dark = dark
    }
    
    init(light: () -> T, dark: () -> U) {
        self.light = light()
        self.dark = dark()
    }
    
    @ViewBuilder var body: some View {
        if colorScheme == .light {
            light
        } else {
            dark
        }
    }
}
