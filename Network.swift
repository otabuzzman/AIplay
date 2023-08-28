import Foundation
import Metal

struct Network {
    private var layers: [Layer]
    private var alpha: Float
    
    // each layer's output O
    private var O: [Matrix<Float>]!
    
    init(_ layers: [Layer], alpha: Float) {
        self.layers = layers
        self.alpha = alpha
    }
    
    mutating func query(for I: Matrix<Float>) -> Matrix<Float> {
        O = [I] // first is output of pseudo input layer, which corresponds to input data
        for layer in layers { // query each layer in turn with output of previous
            O.append(layer.query(for: O.last!))
        }
        return O.last! // network's output layer O
    }
    
    mutating func train(for I: Matrix<Float>, with T: Matrix<Float>) -> Void {
        // network error at output layer O as square of difference T - O
        var E = T - query(for: I)
        // back propagate error layer by layer in reverse order
        for layer in (0..<layers.count).reversed() {
            E = layers[layer].train(for: O[layer], with: E, alpha: alpha)
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

fileprivate let activationKernels = """
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(const device float *input [[ buffer(0) ]], device float *result [[ buffer(1) ]], uint id [[ thread_position_in_grid ]]) {
    result[id] = 1.0 / (1.0 + exp(-input[id]));
}
"""

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let activationLibrary = try device.makeLibrary(source: activationKernels, options: nil)

func sigmoid(_ input: [Float]) throws -> [Float] {
    var input = input
    let inputCount = input.count*MemoryLayout<Float>.size
    var result = Array<Float>(repeating: 0, count: inputCount)
    
    let inputBuffer = device.makeBuffer(bytes: &input, length: inputCount)
    let resultBuffer = device.makeBuffer(bytes: &result, length: inputCount)
    
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    computeCommandEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(resultBuffer, offset: 0, index: 1)
    
    let function = activationLibrary.makeFunction(name: "sigmoid")!
    let descriptor = try device.makeComputePipelineState(function: function)
    computeCommandEncoder.setComputePipelineState(descriptor)
    
    let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
    let threadGroupsCount = MTLSizeMake((inputCount + 31) / 32, 1, 1)
    computeCommandEncoder.dispatchThreadgroups(threadGroupsCount, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let typedResult = resultBuffer!.contents().bindMemory(to: Float.self, capacity: inputCount)
    return Array(UnsafeBufferPointer<Float>(start: typedResult, count: inputCount))
}
