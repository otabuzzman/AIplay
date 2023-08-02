import Foundation

class Network: ObservableObject {
    private var layers: [Layer]
    private var alpha: Float // learning rate
    
    // each layer's output O
    private var O: [Matrix<Float>]!
    
    init(_ layers: [Layer], alpha: Float) {
        self.layers = layers
        self.alpha = alpha
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
            E = layers[layer].train(for: O[layer], with: E, alpha)
        }
    }
}

struct Layer {
    private let inputs: Int
    private let punits: Int
    private let f: (Float) -> Float
    
    private var W: Matrix<Float>
    
    init(
        numberOfInputs inputs: Int = 1,
        numberOfPUnits punits: Int = 1, 
        activationFunction f: @escaping (Float) -> Float = { $0 }
    ) {
        self.inputs = inputs
        self.punits = punits
        self.f = f
        
        W = Matrix<Float>(rows: punits, columns: inputs).map { _ in Float.random(in: -0.5...0.5) }
    }
    
    func query(for I: Matrix<Float>) -> Matrix<Float> {
        return (W • I).map { f($0) }
    }
    
    mutating func train(for I: Matrix<Float>, with E: Matrix<Float>, _ alpha: Float) -> Matrix<Float> {
        let O = query(for: I)
        let B = W.T • E
        W = W + alpha * ((E * O * (1.0 - O)) • I.T)
        return B
    }
}
