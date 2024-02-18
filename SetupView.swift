import SwiftUI

extension ActivationFunction: Identifiable {
    var id: Self { self }
}

struct LayerSetupView: View {
    @Binding private var path: NavigationPath // pop to root. not used, but kept it anyway
    private var commit: (LayerConfig) -> Void
    
    @State private var inputs: Int
    @State private var punits: Int
    @State private var f: ActivationFunction
    @State private var tryOnGpu: Bool
    
    var body: some View {
        Form {
            Section {
                HStack {
                    Text("Layer type")
                    Spacer()
                    Text(punits == 0 ? "Input" : "Fully connected")
                        .foregroundStyle(.secondary)
                }
                switch punits {
                case 0:
                    HStack {
                        Text("Input nodes")
                        Spacer()
                        TextField("number", value: $inputs, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                default:
                    HStack {
                        Text("Processing nodes")
                        Spacer()
                        TextField("number", value: $punits, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                    Picker("Activation function", selection: $f) {
                        ForEach(ActivationFunction.allCases) { f in
                            Text("\(f.description)")
                        }
                    }
                    .pickerStyle(.navigationLink)
                    Toggle(isOn: $tryOnGpu, label: {
                        Text("Activation uses GPU")
                    })
                }
            } header: {
                Text(inputs == 1 && punits == 1 ? "ADD LAYER" : "LAYER SETUP")
            }
        }
        .toolbar { 
            Button("Commit") {
                commit(LayerConfig(inputs: inputs, punits: punits, f: f, tryOnGpu: tryOnGpu))
                path = NavigationPath()
            }
        }
    }
}

extension LayerSetupView {
    init(path: Binding<NavigationPath>, _ config: LayerConfig, commit: @escaping (LayerConfig) -> Void) {
        _path = path
        
        _inputs = State(initialValue: config.inputs)
        _punits = State(initialValue: config.punits)
        _f = State(initialValue: config.f)
        _tryOnGpu = State(initialValue: config.tryOnGpu)
        
        self.commit = commit
    }
}

struct NetworkSetupView: View {
    @Binding private var isPresented: Bool
    private let commit: (NetworkConfig) -> Void
    
    @State private var name: String
    @State private var epochsWanted: Int
    @State private var miniBatchSize: Int
    @State private var alpha: Float
    @State private var inputs: LayerConfig
    @State private var layers: [LayerConfig]
    
    @State private var path = NavigationPath()
    
    var body: some View {
        NavigationStack(path: $path) {
            Form {
                Section {
                    HStack(spacing: 0) {
                        Text("File name")
                        Spacer()
                        TextField("name", text: $name)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.asciiCapable)
                            .frame(width: 128)
                        Text(".nnxd")
                            .foregroundStyle(.gray)
                    }
                }
                Section {
                    HStack {
                        Text("Epochs wanted")
                        Spacer()
                        TextField("number", value: $epochsWanted, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                    HStack {
                        Text("Mini-batch size")
                        Spacer()
                        TextField("number", value: $miniBatchSize, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                    HStack {
                        Text("Learning rate")
                        Spacer()
                        TextField("float", value: $alpha, formatter: Self.formatter)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.decimalPad)
                            .frame(width: 96)
                    }
                } header: {
                    Text("HYPER PARAMS")
                        .font(.subheadline)
                }
                Section {
                    NavigationLink(value: inputs) {
                        HStack {
                            Text("Inputs")
                            Spacer()
                            Text("\(inputs.inputs)")
                            Image(systemName: "line.horizontal.3")
                                .foregroundColor(.clear)
                        }
                    }
                    List {
                        // laggy list item move when using array indices instead of State
                        // https://www.reddit.com/r/SwiftUI/comments/16aytl4/comment/jzak15t
                        ForEach($layers) { layer in
                            // how to use value of type Binding
                            // https://stackoverflow.com/a/72584743
                            NavigationLink(value: layer) {
                                HStack {
                                    Text("Fully connected")
                                    Spacer()
                                    Text("\(layer.wrappedValue.punits)")
                                    Image(systemName: "line.horizontal.3")
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                        .onDelete(perform: layers.count > 1 ? { indices in
                            layers.remove(atOffsets: indices)
                        } : nil)
                        .onMove(perform: layers.count > 1 ? { indices, offset in
                            layers.move(fromOffsets: indices, toOffset: offset)
                        } : nil)
                    }
                } header: {
                    HStack {
                        Text("LAYERS")
                            .font(.subheadline)
                        Spacer()
                        NavigationLink(value: LayerConfig(inputs: 1, punits: -1, f: .identity, tryOnGpu: false)) {
                            Image(systemName: "plus")
                        }
                    }
                } footer: {
                    HStack {
                        Spacer()
                        Button(role: .destructive) {
                            $isPresented.wrappedValue.toggle()
                        } label: {
                            Text("Dismiss")
                        }
                        Spacer()
                    }
                    .padding()
                }
            }
            .navigationDestination(for: LayerConfig.self) { config in 
                switch config.punits {
                case 0: // input layer
                    LayerSetupView(path: $path, config, commit: { newConfig in inputs = newConfig })
                case -1: // add layer
                    LayerSetupView(path: $path, config, commit: { newConfig in layers.append(newConfig) })
                default:
                    EmptyView()
                }
            }
            .navigationDestination(for: Binding<LayerConfig>.self) { config in 
                LayerSetupView(path: $path, config.wrappedValue) { newConfig in 
                    config.wrappedValue = newConfig
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Label("SETUP", systemImage: "gearshape")
                        .labelStyle(SetupLabelStyle())
                        .foregroundStyle(.secondary)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Commit") {
                        commit(config)
                    }
                }
            }
        }
    }
}

extension NetworkSetupView {
    init(isPresented: Binding<Bool>, _ config: NetworkConfig, commit: @escaping (NetworkConfig) -> Void) {
        _isPresented = isPresented
        
        _name = State(initialValue: config.name)
        _epochsWanted = State(initialValue: config.epochsWanted)
        _miniBatchSize = State(initialValue: config.miniBatchSize)
        _alpha = State(initialValue: config.alpha)
        _inputs = State(initialValue: config.inputs)
        _layers = State(initialValue: config.layers)
        
        self.commit = commit
    }
}

extension NetworkSetupView {
    private static var formatter: NumberFormatter {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.decimalSeparator = "."
        formatter.groupingSeparator = ""
        return formatter
    }
    
    private var config: NetworkConfig {
        var layers = [LayerConfig(inputs: inputs.inputs, punits: self.layers[0].punits, f: self.layers[0].f, tryOnGpu: self.layers[0].tryOnGpu)]
        for index in 1..<self.layers.count {
            layers.append(LayerConfig(inputs: layers[index - 1].punits, punits: self.layers[index].punits, f: self.layers[index].f, tryOnGpu: self.layers[index].tryOnGpu))
        }
        return NetworkConfig(name: name, epochsWanted: epochsWanted, miniBatchSize: miniBatchSize, alpha: alpha, inputs: inputs, layers: layers)
    }
}

extension Binding: Equatable where Value: Equatable {
    public static func ==(lhs: Binding<Value>, rhs: Binding<Value>) -> Bool {
        lhs.wrappedValue == rhs.wrappedValue
    }
}

extension Binding: Hashable where Value: Hashable {
    public func hash(into hasher: inout Hasher) {
        self.wrappedValue.hash(into: &hasher)
    }
}

// https://developer.apple.com/forums/thread/666678
struct SetupLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack {
            configuration.icon.font(.headline)
            configuration.title.font(.headline)
        }
    }
}
