import SwiftUI

extension Activation: Identifiable {
    var id: Self { self }
}

struct LayerSetupView: View {
    @Binding var path: NavigationPath // pop to root. not used, but kept it anyway
    @State var config: LayerConfig
    var commit: (LayerConfig) -> Void
        
    var body: some View {
        Form {
            Section {
                HStack {
                    Text("Layer type")
                    Spacer()
                    Text(config.punits == 0 ? "Input" : "Fully connected")
                        .foregroundStyle(.secondary)
                }
                switch config.punits {
                case 0:
                    HStack {
                        Text("Input nodes")
                        Spacer()
                        TextField("number", value: $config.inputs, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                default:
                    HStack {
                        Text("Processing nodes")
                        Spacer()
                        TextField("number", value: $config.punits, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                    Picker("Activation function", selection: $config.f) {
                        ForEach(Activation.allCases) { f in
                            Text("\(f.description)")
                        }
                    }
                    .pickerStyle(.navigationLink)
                    Toggle(isOn: $config.tryOnGpu, label: {
                        Text("Activation uses GPU")
                    })
                }
            } header: {
                Text(config.inputs == 1 && config.punits == 1 ? "ADD LAYER" : "LAYER SETUP")
            }
        }
        .toolbar { 
            Button("Commit") {
                commit(config)
                path = NavigationPath()
            }
        }
    }
}

struct NetworkSetupView: View {
    @Binding var isPresented: Bool
    @Binding var name: String
    @Binding var epochsWanted: Int
    @State var config: NetworkConfig
    var commit: (NetworkConfig) -> Void
    
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
                    HStack {
                        Text("Epochs wanted")
                        Spacer()
                        TextField("number", value: $epochsWanted, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                }
                Section {
                    HStack {
                        Text("Mini-batch size")
                        Spacer()
                        TextField("number", value: $config.miniBatchSize, format: .number)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 96)
                    }
                    HStack {
                        Text("Learning rate")
                        Spacer()
                        TextField("float", value: $config.alpha, formatter: Self.formatter)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.decimalPad)
                            .frame(width: 96)
                    }
                } header: {
                    Text("HYPER PARAMS")
                        .font(.subheadline)
                }
                Section {
                    NavigationLink(value: config.inputs) {
                        HStack {
                            Text("Inputs")
                            Spacer()
                            Text("\(config.inputs.inputs)")
                            Image(systemName: "line.horizontal.3")
                                .foregroundColor(.clear)
                        }
                    }
                    List {
                        // laggy list item move when using array indices instead of State
                        // https://www.reddit.com/r/SwiftUI/comments/16aytl4/comment/jzak15t
                        ForEach($config.layers) { layer in
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
                        .onDelete(perform: config.layers.count > 1 ? { indices in
                            config.layers.remove(atOffsets: indices)
                        } : nil)
                        .onMove(perform: config.layers.count > 1 ? { indices, offset in
                            config.layers.move(fromOffsets: indices, toOffset: offset)
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
                    LayerSetupView(path: $path, config: config) { newConfig in
                        self.config.inputs = newConfig
                    }
                case -1: // add layer
                    LayerSetupView(path: $path, config: config) { newConfig in
                        self.config.layers.append(newConfig)
                    }
                default:
                    EmptyView()
                }
            }
            .navigationDestination(for: Binding<LayerConfig>.self) { config in 
                LayerSetupView(path: $path, config: config.wrappedValue) { newConfig in 
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
                        commit(compile)
                    }
                }
            }
        }
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
    
    private var compile: NetworkConfig {
        var inputs = config.layers[0]
        inputs.inputs = config.inputs.inputs
        var layers = [inputs]
        for index in 1..<config.layers.count {
            var layer = config.layers[index]
            layer.inputs = config.layers[index - 1].punits
            layers.append(layer)
        }
        return NetworkConfig(miniBatchSize: config.miniBatchSize, alpha: config.alpha, inputs: config.inputs, layers: layers)
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
