import SwiftUI
import PencilKit

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
    
    @State private var epochsWanted: Int = 0
    @State private var miniBatchSize: Int = 0
    
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
                                    let measures = await viewModel.train(miniBatchSize: miniBatchSize)
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
                                let data = encodeNNXD()
                                document = NetworkExchangeDocument(data)
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
                    let config = try decodeNNXD(contentsOf: url[0]) { network, measures in
                        viewModel.network = network.pointee
                        self.measures = measures.pointee
                    }
                    epochsWanted = config.epochsWanted
                    miniBatchSize = config.miniBatchSize
                    setNetworkConfig(config)
                } catch {
                    self.error = error as? NetworkExchangeError
                }
            case .failure(let error):
                self.error = .nnxdLoad(error)
            }
        }
        // modifier attached to Button reopens system interface on pressing return and hide keys
        .fileExporter(isPresented: $isExporting, document: document, contentType: .nnxd, defaultFilename: getNetworkConfig()?.name ?? "Untitled") { result in
            switch result {
            case .success:
                break
            case .failure(let error):
                self.error = .nnxdSave(error)
            }
        }
        .sheet(isPresented: $showSetupView) {
            let networkConfig = getNetworkConfig() ?? .default
            NetworkSetupView(isPresented: $showSetupView, networkConfig) { newConfig in
                viewModel.network = GenericFactory.create(NetworkFactory(), newConfig)
                setNetworkConfig(newConfig)
                
                showSetupView.toggle()
                
                if _isDebugAssertConfiguration() {
                    let _ = print(newConfig)
                }
            }
        }
        .task {
            let storedConfig = getNetworkConfig() ?? .default
            guard
                let model = Bundle.main.url(forResource: storedConfig.name, withExtension: "nnxd")
            else { return }
            let loadedConfig = try! decodeNNXD(contentsOf: model) { network, measures in
                viewModel.network = network.pointee
                self.measures = measures.pointee
            }
            epochsWanted = loadedConfig.epochsWanted
            miniBatchSize = loadedConfig.miniBatchSize
        }
    }
}

extension NetworkView {
    private func encodeNNXD() -> Data {
        // header
        var data = nnxdMagic.data(using: .utf8)! // cannot use .encode here
        data += nnxdVersion.encode
        
        // section: hyper parameters
        data += epochsWanted.encode
        data += miniBatchSize.encode
        
        // section: network
        data += viewModel.network.encode
        
        // section: measures
        data += measures.count.encode
        measures.forEach { data += $0.encode }
        
        return data
    }
    
    private func decodeNNXD(contentsOf: URL, loader: ((UnsafeMutablePointer<Network>, UnsafeMutablePointer<Array<Measures>>) -> Void)? = nil) throws -> NetworkConfig {
        var data: Data
        
        do { // read NNXD
            data = try Data(contentsOf: contentsOf)
        } catch { throw NetworkExchangeError.nnxdRead(contentsOf, error) }
        
        // header
        guard // check magic string... (cannot use .init? here)
            let magic = String(data: data[..<nnxdMagic.count], encoding: .utf8)
        else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        
        if magic != nnxdMagic { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        data = data.advanced(by: nnxdMagic.count)
        
        guard // ...and version
            let version = Int(from: data)
        else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        
        if version != nnxdVersion { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        // section: hyper parameters
        guard let epochsWanted = Int(from: data) else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        data = data.advanced(by: MemoryLayout<Int>.size)
        guard let miniBatchSize = Int(from: data) else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        // section: network
        guard let networkSize = Int(from: data) else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var network = Network(from: data)
        if network == nil { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        data = data.advanced(by: networkSize)
        
        // section: measures
        guard let measuresCount = Int(from: data) else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var measures = [Measures]()
        for _ in 0..<measuresCount {
            guard let measureSize = Int(from: data) else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
            data = data.advanced(by: MemoryLayout<Int>.size)
            guard let element = Measures(from: data) else { throw NetworkExchangeError.nnxdDecode(contentsOf) }
            data = data.advanced(by: measureSize)
            measures.append(element)
        }
        
        var networkConfig = network!.config
        networkConfig.name = contentsOf
            .deletingPathExtension()
            .lastPathComponent
        networkConfig.epochsWanted = epochsWanted
        networkConfig.miniBatchSize = miniBatchSize
        
        loader?(&network!, &measures)
        return networkConfig
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
