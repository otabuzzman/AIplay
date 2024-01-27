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
    
    private func query(for I: Matrix<Float>, _ O: inout [Matrix<Float>]) -> Matrix<Float> {
        O.append(I) // first is output of pseudo input layer, which corresponds to input data
        for layer in layers { // query each layer in turn with output of previous
            O.append(layer.query(for: O.last!))
        }
        return O.last! // network's output layer O
    }
    
    // train network with single input vector
    mutating func train(for I: Matrix<Float>, with T: Matrix<Float>) -> Void {
        // each layer's output
        var O: [Matrix<Float>] = []
        // network error at output layer O as square of difference T - O
        var E = T - query(for: I, &O)
        // back propagate error layer by layer in reverse order
        for layer in (0..<layers.count).reversed() {
            E = layers[layer].train(for: O[layer], O[layer + 1], with: E, alpha: alpha)
        }
    }
    
    // train network with multiple input vectors (batch)
    mutating func train(for I: [Matrix<Float>], with T: [Matrix<Float>]) async -> Void {
        assert(I.count == T.count && I.count > 1, "different batchsizes for I and T")
        var E = T[0] - query(for: I[0])
        for index in 1..<I.count - 1 {
            E = (E + (T[index] - query(for: I[index]))) / 2
        }
        // each layer's output
        var O: [Matrix<Float>] = []
        E = (E + (T[I.count - 1] - query(for: I[I.count - 1], &O))) / 2
        for layer in (0..<layers.count).reversed() {
            E = layers[layer].train(for: O[layer], O[layer + 1], with: E, alpha: alpha)
        }
    }
}

extension Network: CustomStringConvertible {
    var description: String {
        "Network(layers: \(layers), alpha: \(alpha))"
    }
}

extension Network: CustomCoder {
    static let magicNumber = "!NNXD"
    
    init?(from: Data) {
        guard
            String(from: from.subdata(in: 0..<Self.magicNumber.count)) == Self.magicNumber
        else { return nil }
        var data = from.advanced(by: Self.magicNumber.count)
        
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
        var data = Self.magicNumber.encode
        data += alpha.encode
        data += layers.count.encode
        layers.forEach { data += $0.encode }
        return data
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

    mutating func train(for I: Matrix<Float>, _ O: Matrix<Float>, with E: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        let e = W.T • E
        W += alpha * ((E * f.apply(O, derivative: true, tryOnGpu: false)) • I.T)
        return e
    }
}

extension Layer: CustomStringConvertible {
    var description: String {
        "Layer(inputs: \(inputs), punits: \(punits), f: \(f), W: \(W))"
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
    var epochsWanted: Int
    var miniBatchSize: Int
    var alpha: Float
    var inputs: LayerConfig
    var layers: [LayerConfig]
}

extension NetworkConfig {
    init(_ epochsWanted: Int, _ miniBatchSize: Int, _ alpha: Float, _ inputs: LayerConfig, _ layers: [LayerConfig]) {
        self.epochsWanted = epochsWanted
        self.miniBatchSize = miniBatchSize
        self.alpha = alpha
        self.inputs = inputs
        self.layers = layers
    }
}

extension NetworkConfig: CustomStringConvertible {
    var description: String {
        "NetworkConfig(epochsWanted: \(epochsWanted), miniBatchSize: \(miniBatchSize), alpha: \(alpha), inputs: \(inputs.inputs), layers: \(layers)"
    }
}

extension NetworkConfig: CustomCoder {
    var encode: Data {
        var data = epochsWanted.encode
        data += miniBatchSize.encode
        data += alpha.encode
        data += inputs.encode
        data += layers.count.encode
        layers.forEach { data += $0.encode }
        return data
    }
    
    init?(from: Data) {
        var data = from
        
        guard let epochsWanted = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)

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
        
        self.init(epochsWanted, miniBatchSize, alpha, inputs, layers)
    }
}

struct LayerConfig: Identifiable, Hashable {
    var id = UUID()
    var inputs: Int
    var punits: Int
    var f: ActivationFunction
    var tryOnGpu: Bool
}

extension LayerConfig {
    init(_ inputs: Int, _ punits: Int, _ f: ActivationFunction, _ tryOnGpu: Bool) {
        self.inputs = inputs
        self.punits = punits
        self.f = f
        self.tryOnGpu = tryOnGpu
    }
}

extension LayerConfig: CustomStringConvertible {
    var description: String {
        "LayerConfig(inputs: \(inputs), punits: \(punits), f: \(f), tryOnGpu: \(tryOnGpu)"
    }
}

extension LayerConfig: CustomCoder {
    var encode: Data {
        var data = inputs.encode
        data += punits.encode
        data += f.rawValue.encode
        data += tryOnGpu.encode
        return data.count.encode + data
    }
    
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
        
        self.init(inputs, punits, f, tryOnGpu)
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
