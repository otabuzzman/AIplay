import SwiftUI
import Foundation

struct Network: View {
    @ObservedObject var viewModel: NetworkViewModel
    
    var body: some View {
        VStack {
            Circle().foregroundColor(viewModel.state.color)
            Button {
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
        }
    }
}

enum NetworkState {
    case random
    case stained
    case trained
    
    var color: Color {
        switch self {
        case .random:
            return .gray
        case .stained:
            return .yellow
        case .trained:
            return .green
        }
    }
}

final class NetworkViewModel: ObservableObject {
    private var layers: [Layer]
    private var learningRate: Float
    
    @Published var state: NetworkState = .random
    
    // each layer's output O
    private var O: [Matrix<Float>]!
    
    init(_ layers: [Layer], learningRate: Float) {
        self.layers = layers
        self.learningRate = learningRate
    }
    
    func query(for I: Matrix<Float>) -> Matrix<Float> {
        O = [I] // first is output of pseudo input layer, which corresponds to input data
        for layer in layers { // query each layer in turn with output of previous
            O.append(layer.query(for: O.last!))
        }
        return O.last! // network's output layer O
    }
    
    func train(for I: Matrix<Float>, with T: Matrix<Float>) -> Void {
        // network error at output layer O as square of difference T - O
        var E = T - query(for: I)
        // back propagate error layer by layer in reverse order
        for layer in (0..<layers.count).reversed() {
            E = layers[layer].train(for: O[layer], with: E, alpha: learningRate)
        }
        state = .trained
    }
}

extension NetworkViewModel: CustomStringConvertible {
    var description: String {
        "NetworkViewModel(layers: \(layers), learningRate: \(learningRate))"
    }
}

extension NetworkViewModel: CustomCodable {
    static let magicNumber = "!NN"
    
    convenience init?(from: Data) {
        guard
            String(from: from.subdata(in: 0..<Self.magicNumber.count)) == Self.magicNumber
        else { return nil }
        var data = from.advanced(by: Self.magicNumber.count)
        
        guard let learningRate = Float(from: data)?.bigEndian else { return nil }
        data = data.advanced(by: MemoryLayout<Float>.size)
        
        guard let layersCount = Int(from: data)?.bigEndian else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var layers = [Layer]()
        for _ in 0..<layersCount {
            guard let layerSize = Int(from: data)?.bigEndian else { return nil }
            data = data.advanced(by: MemoryLayout<Int>.size)
            guard let layer = Layer(from: data) else { return nil }
            layers.append(layer)
            data = data.advanced(by: layerSize)
        }
        
        self.init(layers, learningRate: learningRate)
    }
    
    func encode() throws -> Data {
        var data = Self.magicNumber.encode
        data += learningRate.bigEndian.encode
        data += layers.count.bigEndian.encode
        try layers.forEach { data += try $0.encode() }
        return data
    }
}

enum ActivationFunction: Int {
    case identity = 1
    case sigmoid
    
    var implementation: (Float) -> Float {
        switch self {
        case .identity:
            return { x in x }
        case .sigmoid:
            return { x in 1.0 / (1.0 + expf(-x)) }
        }
    }
}

struct Layer {
    private let inputs: Int
    private let punits: Int
    private let f: ActivationFunction
    private var W: Matrix<Float>
    
    init(
        numberOfInputs inputs: Int = 1,
        numberOfPUnits punits: Int = 1,
        activationFunction f: ActivationFunction = .identity,
        weights W: Matrix<Float>? = nil
    ) {
        self.inputs = inputs
        self.punits = punits
        self.f = f
        
        if let W = W {
            self.W = W
        } else {
            self.W = Matrix<Float>(rows: punits, columns: inputs).map { _ in Float.random(in: -0.5...0.5) }
        }
    }
    
    func query(for I: Matrix<Float>) -> Matrix<Float> {
        return (W • I).map { f.implementation($0) }
    }
    
    mutating func train(for I: Matrix<Float>, with E: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        let O = query(for: I)
        let B = W.T • E
        W += alpha * ((E * O * (1.0 - O)) • I.T)
        return B
    }
}

extension Layer: CustomStringConvertible {
    var description: String {
        "Layer(inputs: \(inputs), punits: \(punits), f: \(f), W: \(W))"
    }
}

extension Layer: CustomCodable {
    init?(from: Data) {
        var data = from
        
        guard let inputs = Int(from: data)?.bigEndian else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let punits = Int(from: data)?.bigEndian else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)

        guard
            let activationFunction = Int(from: data)?.bigEndian,
            let f = ActivationFunction(rawValue: activationFunction)
        else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let W = Matrix<Float>(from: data) else { return nil }
        
        self.init(numberOfInputs: inputs, numberOfPUnits: punits, activationFunction: f, weights: W)
    }
    
    func encode() throws -> Data {
        var data = inputs.bigEndian.encode
        data += punits.bigEndian.encode
        data += f.rawValue.bigEndian.encode
        data += try W.encode()
        return data.count.bigEndian.encode + data
    }
}
