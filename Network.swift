import Foundation
import Metal

struct Network {
    private var layers: [Layer]
    private var alpha: Float
    
    // each layer's output O in batch (the outer array)
    private var O: [[Matrix<Float>]]!
    
    init(_ layers: [Layer], alpha: Float) {
        self.layers = layers
        self.alpha = alpha
    }
    
    // let network predict output for single input vector
    mutating func query(for I: Matrix<Float>) -> Matrix<Float> {
        O = [[I]] // first is output of pseudo input layer, which corresponds to input data
        for layer in layers { // query each layer in turn with output of previous
            O[0].append(layer.query(for: O.first!.last!))
        }
        return O.first!.last! // network's output layer O
    }
    
    // let network predict outputs for multiple input vectors (batch)
    mutating func query(for I: [Matrix<Float>]) -> [Matrix<Float>] {
        assert(I.count > 0, "empty batch in I")
        O = [I]
        for layer in layers {
            O.append(layer.query(for: O.last!))
        }
        return O.last!
    }
    
    // train network with single input vector
    mutating func train(for I: Matrix<Float>, with T: Matrix<Float>) -> Void {
        // network error at output layer O as square of difference T - O
        var E = T - query(for: I)
        // back propagate error layer by layer in reverse order
        for layer in (0..<layers.count).reversed() {
            E = layers[layer].train(for: O.first![layer], with: E, alpha: alpha)
        }
    }
    
    // train network with multiple input vectors (batch)
    mutating func train(for I: [Matrix<Float>], with T: [Matrix<Float>]) async -> Void {
        assert(I.count == T.count && I.count > 0, "different batchsizes for I and T")
        let o = query(for: I)
        var E = I.enumerated().map { index, value in T[index] - o[index] }
        for layer in (0..<layers.count).reversed() {
            E = await layers[layer].train(for: O[layer], with: E, alpha: alpha)
        }
    }
}

extension Network: CustomStringConvertible {
    var description: String {
        "NetworkViewModel(layers: \(layers), alpha: \(alpha))"
    }
}

extension Network: CustomCoder {
    static let magicNumber = "!NNXD"
    
    init?(from: Data) {
        guard
            String(from: from.subdata(in: 0..<Self.magicNumber.count)) == Self.magicNumber
        else { return nil }
        var data = from.advanced(by: Self.magicNumber.count)
        
        guard let alpha = Float(from: data)?.bigEndian else { return nil }
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
        
        self.init(layers, alpha: alpha)
    }
    
    var encode: Data {
        var data = Self.magicNumber.encode
        data += alpha.bigEndian.encode
        data += layers.count.bigEndian.encode
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
            self.W = Matrix<Float>(rows: punits, columns: inputs).map { _ in Float.random(in: -0.5...0.5) }
        }
    }
    
    func query(for I: Matrix<Float>) -> Matrix<Float> {
        f.implementation(W • I, tryOnGpu: false)
    }
    
    func query(for I: [Matrix<Float>]) -> [Matrix<Float>] {
        assert(I.count > 0, "empty batch in I")
        return I.map { f.implementation(W • $0, tryOnGpu: false) }
    }

    mutating func train(for I: Matrix<Float>, with E: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        let e = W.T • E
        let o = query(for: I)
        W += alpha * ((E * o * (1.0 - o)) • I.T)
        return e
    }
    
    mutating func train(for I: [Matrix<Float>], with E: [Matrix<Float>], alpha: Float) async -> [Matrix<Float>] {
        assert(I.count == E.count && I.count > 0, "different batchsizes for I and E")
        let e = E.map { W.T • $0 } // back propagation error (e) according to this layer's error (E)
        let g = await withTaskGroup(of: Matrix<Float>.self) { batch in // batch gradient (g)
            for i in 0..<I.count {
                let o = query(for: I[i]) // this layer's prediction (o) for input (I[i])
                batch.addTask {
                    return (E[i] * o * (1.0 - o)) • I[i].T // this input's (I[i]) gradient
                }
            }
            var g = await batch.next()! // first gradient
            for await result in batch { // remaining gradients
                g = (g + result) / 2 // arithmetic mean cumulation
            }
            return g
        }
        W += alpha * g
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
    
    var encode: Data {
        var data = inputs.bigEndian.encode
        data += punits.bigEndian.encode
        data += f.rawValue.bigEndian.encode
        data += W.encode
        return data.count.bigEndian.encode + data
    }
}

typealias NetworkConfig = (
    layersWithSizes: [Int], activationFunctions: [ActivationFunction], learningRate: Float
)

struct NetworkFactory: AbstractFactory {
    func create(_ config: NetworkConfig) -> Network? {
        guard
            config.layersWithSizes.count > 1,
            config.layersWithSizes.count - 1 == config.activationFunctions.count
        else { return nil }
        var layers: [Layer] = []
        for index in 1..<config.layersWithSizes.count {
            let prevLayerSize = config.layersWithSizes[index - 1]
            let thisLayerSize = config.layersWithSizes[index]
            let layer = Layer(
                numberOfInputs: prevLayerSize,
                numberOfPUnits: thisLayerSize,
                activationFunction: config.activationFunctions[index - 1])
            layers.append(layer)
        }
        return Network(layers, alpha: config.learningRate)
    }
}

enum ActivationFunction: Int {
    case identity = 1
    case sigmoid
}

extension ActivationFunction {
    func implementation(_ input: Matrix<Float>, tryOnGpu gpu: Bool = false) -> Matrix<Float> {
        let fallback = Self.impl4Cpu[self.rawValue]
        return gpu ? Self.impl4Gpu(self, input) ?? fallback(input) : Self.impl4Cpu[self.rawValue](input)
    }
    
    private static let impl4Cpu: [(Matrix<Float>) -> Matrix<Float>] = [
        { dummy in dummy },
        { input in input }, // identity
        { input in input.map { 1.0 / (1.0 + expf(-$0)) } } // sigmoid
    ]
    
    private static func impl4Gpu(_ f: ActivationFunction, _ input: Matrix<Float>) -> Matrix<Float>? {
        var result: Matrix<Float>?
        if let entries = try? activationKernel(f, input.entries) {
            result = Matrix<Float>(rows: input.rows, columns: input.columns, entries: entries)
        }
        return result
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
"""

fileprivate let device = MTLCreateSystemDefaultDevice()
fileprivate let queue = device?.makeCommandQueue()
fileprivate let library = try? device?.makeLibrary(source: activationLibrary, options: nil)

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

fileprivate func activationKernel(_ function: ActivationFunction, _ input: [Float]) throws -> [Float] {
    var input = input
    let inputCount = input.count * MemoryLayout<Float>.size
    var result = Array<Float>(repeating: 0, count: inputCount)
    
    guard
        let inputBuffer = device?.makeBuffer(bytes: &input, length: inputCount),
        let resultBuffer = device?.makeBuffer(bytes: &result, length: inputCount)
    else {
        throw ActivationKernelError.apiReturnedNil("makeBuffer")
    }
    
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
    
    guard
        let function = library?.makeFunction(name: "\(function)")
    else {
        throw ActivationKernelError.apiReturnedNil("makeFunction")
    }
    
    do {
        let descriptor = try device!.makeComputePipelineState(function: function)
        commandEncoder.setComputePipelineState(descriptor)
    } catch {
        throw ActivationKernelError.apiException("makeComputePipelineState", error)
    }
    
    let threadsWidth = 32
    let threadsCount = MTLSizeMake(threadsWidth, 1, 1)
    let groupsWidth = (inputCount + threadsWidth - 1) / threadsWidth
    let groupsCount = MTLSizeMake(groupsWidth, 1, 1)
    commandEncoder.dispatchThreadgroups(groupsCount, threadsPerThreadgroup: threadsCount)
    
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let typedResult = resultBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
    return Array(UnsafeBufferPointer<Float>(start: typedResult, count: input.count))
}
