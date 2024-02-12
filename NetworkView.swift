import SwiftUI
import PencilKit

import UniformTypeIdentifiers

struct NetworkView: View {
    @ObservedObject private var viewModel = NetworkViewModel()
    
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
    
    @State private var batchesTrained = 0
    @State private var measures = [Measures]()
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
                    let result = viewModel.query(image: queryInput)
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
                                let result = viewModel.query(image: queryInput)
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
                                    let range = batchesTrained..<batchesTrained + 1
                                    _ = await viewModel.train(batch: range)
                                    batchesTrained += 1
                                    longRunBusy = false
                                    Task {
                                        try await Task.sleep(nanoseconds: 1_000_000_000)
                                        viewModel.progress = 0
                                    }
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
                                    let measures = await viewModel.train()
                                    self.measures.append(measures)
                                    longRunBusy = false
                                    Task {
                                        try await Task.sleep(nanoseconds: 1_000_000_000)
                                        viewModel.progress = 0
                                    }
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
                                    validationAccuracy = await viewModel.query()
                                    longRunBusy = false
                                    Task {
                                        try await Task.sleep(nanoseconds: 1_000_000_000)
                                        viewModel.progress = 0
                                    }
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
                                document = NetworkExchangeDocument(viewModel.network.encode + measures.encode)
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
                                
                                queryInput = []
                                queryTarget = nil
                                withAnimation() {
                                    showResultDetails = false
                                }
                                resultDetails = nil
                                resultReading = nil
                                batchesTrained = 0
                                measures = []
                                validationAccuracy = nil
                                
                                viewModel.reset()
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
                    var content = try Data(contentsOf: url[0])
                    
                    guard
                        String(from: content.subdata(in: 0..<NetworkExchangeDocument.magic.count)) == NetworkExchangeDocument.magic
                    else {
                        self.error = .nndxDecode(url[0])
                        return
                    }
                    var content = content.advanced(by: Self.magicNumber.count)
                    
                    guard let networkSize = Int(from: content) else { return nil }
                    content = content.advanced(by: MemoryLayout<Int>.size)
                    
                    guard
                        let network = Network(from: content)
                    else {
                        self.error = .nndxDecode(url[0])
                        return
                    }
                    viewModel.network = network
                    content = content.advanced(by: networkSize)
                    
                    guard let measureSize = Int(from: content) else { return nil }
                    content = content.advanced(by: MemoryLayout<Int>.size)

                    guard
                        let measures = Measures(from: content)
                    else {
                        self.error = .nndxDecode(url[0])
                        return
                    }
                    viewModel.measures = measures
                    content = content.advanced(by: measureSize)
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
            let networkConfig = getNetworkConfig() ?? .default
            NetworkSetupView(isPresented: $showSetupView, networkConfig) { newConfig in
                _ = viewModel.setup(config: newConfig)
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
        var network: Network!
        private(set) var dataset: MNISTViewModel!
        private(set) var epochsWanted: Int!
        private(set) var miniBatchSize: Int!
        
        @Published var progress: Float = 0 // 0...1
        
        init(config: NetworkConfig? = nil) {
            if let config = config {
                _ = setup(config: config)
            } else {
                reset()
            }
            dataset = MNISTViewModel()
        }
        
        func reset() -> Void {
            let model = Bundle.main.url(forResource: "default-model", withExtension: "nndx")!
            network = try! Network(from: Data(contentsOf: model))! // should not fail
            
            _ = setup(config: .default)
        }
        
        func setup(config: NetworkConfig) -> Bool {
            guard
                let network = GenericFactory.create(NetworkFactory(), config)
            else { return false }
            self.network = network
            
            epochsWanted = config.epochsWanted
            miniBatchSize = config.miniBatchSize
            
            return true
        }
        
        func query(subset: MNISTSubset.Purpose = .test) async -> Float {
            let samples = dataset.count(in: subset)
            var results = [Int](repeating: 0, count: samples)
            
            for sample in 0..<samples {
                let (image, label) = dataset.fetch(sample, from: subset)
                let result = query(image: image).maxElementIndex()
                results[sample] = result ?? -1 == label ? 1 : 0
                
                let current = Float(sample) / Float(samples)
                advanceProgress(current)
                
                do {
                    try Task.checkCancellation()
                } catch { return 0 }
            }
            
            let accuracy = Float(results.reduce(0, +)) / Float(samples)
            return accuracy
        }
        
        func query(image index: Int) -> [Float] {
            let (image, _) = dataset.fetch(index, from: .test)
            return query(image: image)
        }
        
        func query(image: MNISTImage) -> [Float] {
            network.query(for: image.toInput()).entries
        }
        
        func train() async -> Measures {
            let measures = Measures()
            measures.trainingStartTime = Date.timeIntervalSinceReferenceDate
            defer {
                measures.trainingDuration = Date.timeIntervalSinceReferenceDate - measures.trainingStartTime
            }
            
            let batches = dataset.count(in: .train) / miniBatchSize
            measures.trainingLoss = .init(repeating: 0, count: batches)
            
            for batch in 0..<batches {
                let start = batch * miniBatchSize
                let end = start + miniBatchSize
                let cost = await train(batch: start..<end)
                measures.trainingLoss?[batch] = cost
                
                let current = Float(batch) / Float(batches)
                advanceProgress(current)
                
                do {
                    try Task.checkCancellation()
                } catch { return measures }
            }
            
            measures.trainingAccuracy = await query(subset: .train)
            measures.validationAccuracy = await query(subset: .test)
            
            if _isDebugAssertConfiguration() {
                let _ = print(measures)
            }
            
            return measures
        }
        
        func train(batch range: Range<Int>) async -> Float {
            let (images, labels) = dataset.fetch(range, from: .train)
            let cost = await network.train(for: images.toMatrix(), with: labels.toMatrix())
            return cost
        }
        
        func train(image index: Int) async -> Float {
            let (image, label) = dataset.fetch(index, from: .train)
            let loss = await network.train(for: image.toInput(), with: label.toTarget())
            return loss
        }
        
        private func advanceProgress(_ current: Float) -> Void {
            if current - progress > 0.01 {
                progress += current
            }
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

extension Measures {
    init(_ trainingStartTime: TimeInterval, _ trainingDuration: TimeInterval, _ trainingAccuracy: Float, _ validationAccuracy: Float, _ trainingLoss: [Float]) {
        self.trainingStartTime = trainingStartTime
        self.trainingDuration = trainingDuration
        self.trainingAccuracy = trainingAccuracy
        self.validationAccuracy = validationAccuracy
        self.trainingLoss = trainingLoss
    }
}

extension Measures: CustomStringConvertible {
    var description: String {
        "Measures(trainingStartTime: \(trainingStartTime), trainingDuration: \(trainingDuration), trainingAccuracy: \(trainingAccuracy), validationAccuracy: \(validationAccuracy), trainingLoss: \(trainingLoss == nil ? "nil" : "[\(stringOfElements(in: trainingLoss!, count: 16, format: { String(describing: $0) }))])")"
    }
}

extension Measures: CustomCoder {
    init?(from: Data) {
        var data = from
        
        guard let trainingStartTime = TimeInterval(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<TimeInterval>.size)
        
        guard let trainingDuration = TimeInterval(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<TimeInterval>.size)
        
        guard let trainingAccuracy = Float(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Float>.size)
        
        guard let validationAccuracy = Float(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Float>.size)
        
        guard let layerConfigSize = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        guard let inputs = LayerConfig(from: data) else { return nil }
        data = data.advanced(by: layerConfigSize)
        
        guard let trainingLossCount = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var trainingLoss = [Float]()
        for _ in 0..<trainingLossCount {
            guard let loss = Float(from: data) else { return nil }
            data = data.advanced(by: MemoryLayout<Float>.size)
            trainingLoss.append(loss)
        }
        
        self.init(trainingStartTime, trainingDuration, trainingAccuracy, validationAccuracy, trainingLoss)
    }
    
    var encode: Data {
        var data = trainingStartTime.encode
        data += trainingDuration.encode
        data += trainingAccuracy.encode
        data += validationAccuracy.encode
        data += trainingLoss.count.encode
        trainingLoss.forEach { data += $0.encode }
        return data.count.encode + data
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
    static let magic = "!NNXD"
    
    static var readableContentTypes: [UTType] { [.nnxd] }
    static var writableContentTypes: [UTType] { [.nnxd] }
    
    private(set) var content: Data
    
    init(_ content: Data) {
        self.content = magic.encode + content
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
