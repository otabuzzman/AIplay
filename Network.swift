import Foundation
import GameplayKit
import Metal

struct Network {
    private var layers: [Layer]
    private var alpha: Float
    
    init(_ layers: [Layer], alpha: Float) {
        self.layers = layers
        self.alpha = alpha
    }
    
    // let network predict output for single input vector
    func query(for I: Matrix<Float>) -> Matrix<Float> {
        var O = I // first is output of pseudo input layer, which corresponds to input data
        for layer in layers { // query each layer in turn with output of previous
            O = layer.query(for: O)
        }
        return O // network's output layer O
    }
    
    // predict for single input vector and return each layer’s output
    private func query(for I: Matrix<Float>, _ O: inout [Matrix<Float>]) -> Matrix<Float> {
        O.append(I) // first is output of pseudo input layer, which corresponds to input data
        for layer in layers { // query each layer in turn with output of previous
            O.append(layer.query(for: O.last!))
        }
        return O.last! // network's output layer O
    }
    
    func loss(for I: Matrix<Float>, with T: Matrix<Float>) -> Float {
        let E = T - query(for: I)
        let loss = E.entries.map { v in pow(v, 2) }.reduce(0, +)
        return loss
    }
    
    func cost(for I: [Matrix<Float>], with T: [Matrix<Float>]) async -> Float {
        var E = T[0] - query(for: I[0])
        var C = E.map { v in pow(v, 2) }
        for index in 1..<I.count {
            let e = T[index] - query(for: I[index])
            C = (C + e.map { v in pow(v, 2) }) / 2
            E = (E + e) / 2
        }
        let cost = C.entries.reduce(0, +)
        return cost
    }
    
    // train network with single input vector
    mutating func train(for I: Matrix<Float>, with T: Matrix<Float>) async -> Float {
        // each layer's output
        var O: [Matrix<Float>] = []
        // network error at output layer O as difference of T - O
        var E = T - query(for: I, &O)
        // training loss (MSE L2)
        let loss = E.entries.map { v in pow(v, 2) }.reduce(0, +)
        // back propagate error and update weights layer by layer in reverse order
        for layer in (0..<layers.count).reversed() {
            /*
             * layer holds indices from last to first layer. these correspond
             * to indices of the output of the previous layer in O,
             * which contains an additional pseudolayer for the network input
             * at index 0, thus adding one to the regular layer indices.
             */
            E = layers[layer].train(for: O[layer], O[layer + 1], E, alpha: alpha)
        }
        return loss
    }
    
    // train network with multiple input vectors (batch)
    mutating func train(for I: [Matrix<Float>], with T: [Matrix<Float>]) async -> Float {
        assert(I.count == T.count, "different batch sizes for I and T")
        // mean layer outputs for batch
        var O: [Matrix<Float>] = []
        // init with 1st prediction for stepwise averaging to work
        _ = query(for: I[0], &O)
        // mean network error for batch
        var E = T[0] - O.last!
        // training cost (mean loss (MSE L2) of batch)
        var C = E.map { v in pow(v, 2) }
        // mean network gradient for batch
        var G = layers.last!.gradient(for: O.lastButOne!, O.last!, E)
        // process batch inputs except first
        for index in 1..<I.count {
            // layer outputs for this input
            var o: [Matrix<Float>] = []
            _ = query(for: I[index], &o)
            // stepwise averaging layer outputs in O
            o.enumerated().forEach { i, v in O[i] = (O[i] + v) / 2 }
            // network error for this input
            let e = T[index] - o.last!
            // update cost
            C = (C + e.map { v in pow(v, 2) }) / 2
            // update mean error in E
            E = (E + e) / 2
            // network gradient for this input
            let g = layers.last!.gradient(for: o.lastButOne!, o.last!, e)
            // update mean gradient in G
            G = (G + g) / 2
        }
        // mean error of last but one layer based on mean network error and gradient for batch
        var e = layers[layers.count - 1].train(with: G, E, alpha: alpha)
        // back propagate mean error and update weights layer by layer in reverse order
        for layer in (0..<layers.count - 1).reversed() {
            e = layers[layer].train(for: O[layer], O[layer + 1], e, alpha: alpha)
        }
        let cost = C.entries.reduce(0, +)
        return cost
    }
}

extension Network: CustomStringConvertible {
    var description: String {
        "Network(layers: \(layers), alpha: \(alpha))"
    }
}

extension Network {
    var config: NetworkConfig {
        var other = [LayerConfig]()
        for index in 0..<layers.count {
            other.append(layers[index].config)
        }
        let input = LayerConfig(inputs: other[0].inputs, punits: 0, f: .identity, tryOnGpu: false)
        return NetworkConfig(name: "", miniBatchSize: -1,
                             alpha: alpha, inputs: input, layers: other)
    }
}

extension Network: CustomCoder {
    init?(from: Data) {
        var data = from
        
        guard let alpha = Float(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Float>.size)
        
        guard let layersCount = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var layers = [Layer]()
        for _ in 0..<layersCount {
            guard let layerSize = Int(from: data) else { return nil }
            data = data.advanced(by: MemoryLayout<Int>.size)
            guard let layer = Layer(from: data) else { return nil }
            data = data.advanced(by: layerSize)
            layers.append(layer)
        }
        
        self.init(layers, alpha: alpha)
    }
    
    var encode: Data {
        var data = alpha.encode
        data += layers.count.encode
        layers.forEach { data += $0.encode }
        return data.count.encode + data
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
            // MYONN, p. 103 ff.
            let d = powf(Float(inputs), -0.5)
            let random = GKRandomSource()
            let normal = GaussianDistribution(randomSource: random, mean: 0, deviation: d)!
            // MYONN, p. 133
            self.W = Matrix<Float>(rows: punits, columns: inputs).map { _ in normal.nextFloat() }
            // MYONN, p. 151 ff.
            // self.W = Matrix<Float>(rows: punits, columns: inputs).map { _ in Float.random(in: -0.5...0.5) }
        }
    }
    
    func query(for I: Matrix<Float>) -> Matrix<Float> {
        f.apply(W • I, tryOnGpu: true)
    }
    
    mutating func train(for I: Matrix<Float>, _ O: Matrix<Float>, _ E: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        train(with: gradient(for: I, O, E), E, alpha: alpha)
    }
    
    mutating func train(with G: Matrix<Float>, _ E: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        let e = W.T • E
        W += alpha * G
        return e
    }
    
    func gradient(for I: Matrix<Float>, _ O: Matrix<Float>, _ E: Matrix<Float>) -> Matrix<Float> {
        E * f.apply(O, derivative: true, tryOnGpu: false) • I.T
    }
}

extension Layer: CustomStringConvertible {
    var description: String {
        "Layer(inputs: \(inputs), punits: \(punits), f: \(f), W: \(W))"
    }
}

extension Layer {
    var config: LayerConfig {
        LayerConfig(inputs: inputs, punits: punits, f: f, tryOnGpu: false)
    }
}

extension Layer: CustomCoder {
    init?(from: Data) {
        var data = from
        
        guard let inputs = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let punits = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let activationFunction = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        let f = ActivationFunction(rawValue: activationFunction) ?? .identity
        
        guard let W = Matrix<Float>(from: data) else { return nil }
        
        self.init(numberOfInputs: inputs, numberOfPUnits: punits, activationFunction: f, weights: W)
    }
    
    var encode: Data {
        var data = inputs.encode
        data += punits.encode
        data += f.rawValue.encode
        data += W.encode
        return data.count.encode + data
    }
}

enum ActivationFunction: Int, CaseIterable {
    case identity = 1
    case sigmoid
}

extension ActivationFunction {
    var description: String {
        switch self {
        case .identity:
            return "Identity"
        case .sigmoid:
            return "Sigmoid"
        }
    }
}

extension ActivationFunction {
    func apply(_ input: Matrix<Float>, derivative: Bool = false, tryOnGpu gpu: Bool = false) -> Matrix<Float> {
        let fallback = Self.applyOnCpu[self.rawValue]
        return gpu
        ? Self.applyOnGpu(self, input, derivative) ?? fallback(input, derivative)
        : Self.applyOnCpu[self.rawValue](input, derivative)
    }
    
    private static let applyOnCpu: [(Matrix<Float>, Bool) -> Matrix<Float>] = [
        { nouse, _ in nouse },
        { input, _ in input }, // identity
        { input, derivative in input.map { // sigmoid
            derivative
            ? $0 * (1.0 - $0)
            : 1.0 / (1.0 + expf(-$0)) } }
    ]
    
    private static func applyOnGpu(_ f: ActivationFunction, _ input: Matrix<Float>, _ derivative: Bool) -> Matrix<Float>? {
        var result: Matrix<Float>?
        do {
            let entries = try activationKernel(f, input.entries, derivative)
            result = Matrix<Float>(rows: input.rows, columns: input.columns, entries: entries)
        } catch {
            if _isDebugAssertConfiguration() {
                let _ = print(error)
            }
        }
        return result
    }
}

struct NetworkConfig {
    var name: String
    var miniBatchSize: Int
    var alpha: Float
    var inputs: LayerConfig
    var layers: [LayerConfig]
}

extension NetworkConfig: CustomStringConvertible {
    var description: String {
        "NetworkConfig(name: \(name), miniBatchSize: \(miniBatchSize), alpha: \(alpha), inputs: \(inputs.inputs), layers: \(layers))"
    }
}

extension NetworkConfig: CustomCoder {
    init?(from: Data) {
        var data = from
        
        guard let nameSize = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        guard let name = String(data: data[..<nameSize], encoding: .utf8) else { return nil }
        data = data.advanced(by: nameSize)
        
        guard let miniBatchSize = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let alpha = Float(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Float>.size)
        
        guard let layerConfigSize = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        guard let inputs = LayerConfig(from: data) else { return nil }
        data = data.advanced(by: layerConfigSize)
        
        guard let layerConfigCount = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var layers = [LayerConfig]()
        for _ in 0..<layerConfigCount {
            guard let layerConfigSize = Int(from: data) else { return nil }
            data = data.advanced(by: MemoryLayout<Int>.size)
            guard let layerConfig = LayerConfig(from: data) else { return nil }
            data = data.advanced(by: layerConfigSize)
            layers.append(layerConfig)
        }
        self.init(name: name, miniBatchSize: miniBatchSize, alpha: alpha, inputs: inputs, layers: layers)
    }
    
    var encode: Data {
        let utf8 = name.utf8
        var data = utf8.count.encode
        data += utf8
        data += miniBatchSize.encode
        data += alpha.encode
        data += inputs.encode
        data += layers.count.encode
        layers.forEach { data += $0.encode }
        return data
    }
}

struct LayerConfig: Identifiable, Hashable {
    let id = UUID()
    var inputs: Int
    var punits: Int
    var f: ActivationFunction
    var tryOnGpu: Bool
}

extension LayerConfig: CustomStringConvertible {
    var description: String {
        "LayerConfig(inputs: \(inputs), punits: \(punits), f: \(f), tryOnGpu: \(tryOnGpu)"
    }
}

extension LayerConfig: CustomCoder {
    init?(from: Data) {
        var data = from
        
        guard let inputs = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let punits = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let activationFunction = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        let f = ActivationFunction(rawValue: activationFunction) ?? .identity
        
        guard let tryOnGpu = Bool(from: data) else { return nil }
        
        self.init(inputs: inputs, punits: punits, f: f, tryOnGpu: tryOnGpu)
    }
    
    var encode: Data {
        var data = inputs.encode
        data += punits.encode
        data += f.rawValue.encode
        data += tryOnGpu.encode
        return data.count.encode + data
    }    
}

struct NetworkFactory: AbstractFactory {
    func create(_ config: NetworkConfig) -> Network? {
        guard
            config.layers.count > 0
        else { return nil }
        var layers: [Layer] = []
        for index in 0..<config.layers.count {
            let layer = config.layers[index]
            layers.append(Layer(
                numberOfInputs: layer.inputs,
                numberOfPUnits: layer.punits,
                activationFunction: layer.f))
        }
        return Network(layers, alpha: config.alpha)
    }
}

extension Array {
    var lastButOne: Self.Element? {
        get {
            guard
                self.count > 1
            else { return nil }
            return self[self.count - 2]
        }
    }
}



fileprivate let activationLibrary = """
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(
    const device float *input  [[ buffer(0) ]],
          device float *result [[ buffer(1) ]],
                  uint  id     [[ thread_position_in_grid ]]) {
    result[id] = 1.0 / (1.0 + exp(-input[id]));
}
kernel void sigmoid_derivative(
    const device float *input  [[ buffer(0) ]],
          device float *result [[ buffer(1) ]],
                  uint  id     [[ thread_position_in_grid ]]) {
    result[id] = 1.0 * (1.0 - input[id]);
}
"""

fileprivate let device = MTLCreateSystemDefaultDevice()
fileprivate let queue = device?.makeCommandQueue()
fileprivate let library = try? device?.makeLibrary(source: activationLibrary, options: nil)

fileprivate let sigmoid = library?.makeFunction(name: "sigmoid")
fileprivate let sigmoid_derivative = library?.makeFunction(name: "sigmoid_derivative")

// length must equal punits times element stride of largest layer in this regard
fileprivate let inputBuffer = device?.makeBuffer(length: 400)
fileprivate let resultBuffer = device?.makeBuffer(length: 400)

fileprivate enum ActivationKernelError: Error {
    case apiException(String, Error)
    case apiReturnedNil(String)
}

fileprivate extension ActivationKernelError {
    var description: String {
        switch self {
        case .apiException(let api, let error):
            return "Metal API \(api) threw exception \(error)"
        case .apiReturnedNil(let api):
            return "Metal API \(api) returned nil"
        }
    }
}

fileprivate func activationKernel(_ function: ActivationFunction, _ input: [Float], _ derivative: Bool) throws -> [Float] {
    // var input = input
    let inputCount = input.count * MemoryLayout<Float>.size
    // var result = Array<Float>(repeating: 0, count: inputCount)
    /*
     guard
     let inputBuffer = device?.makeBuffer(bytes: &input, length: inputCount),
     let resultBuffer = device?.makeBuffer(bytes: &result, length: inputCount)
     else {
     throw ActivationKernelError.apiReturnedNil("makeBuffer")
     }
     */
    inputBuffer?.contents().copyMemory(from: input, byteCount: inputCount)
    
    guard
        let commandBuffer = queue?.makeCommandBuffer()
    else {
        throw ActivationKernelError.apiReturnedNil("makeCommandBuffer")
    }
    
    guard
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
        throw ActivationKernelError.apiReturnedNil("makeComputeCommandEncoder")
    }
    
    commandEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
    commandEncoder.setBuffer(resultBuffer, offset: 0, index: 1)
    /*
     guard
     let function = library?.makeFunction(name: derivative ? "\(function)_derivative" : "\(function)")
     else {
     throw ActivationKernelError.apiReturnedNil("makeFunction")
     }
     */
    do {
        let descriptor = try device!.makeComputePipelineState(function: derivative ? sigmoid_derivative! : sigmoid!)
        commandEncoder.setComputePipelineState(descriptor)
    } catch {
        throw ActivationKernelError.apiException("makeComputePipelineState", error)
    }
    
    let threadsWidth = input.count
    let threadsCount = MTLSizeMake(threadsWidth, 1, 1)
    let groupsWidth = (input.count + threadsWidth - 1) / threadsWidth
    let groupsCount = MTLSizeMake(groupsWidth, 1, 1)
    commandEncoder.dispatchThreadgroups(groupsCount, threadsPerThreadgroup: threadsCount)
    
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let typedResult = resultBuffer!.contents().bindMemory(to: Float.self, capacity: input.count)
    return Array(UnsafeBufferPointer<Float>(start: typedResult, count: input.count))
}
