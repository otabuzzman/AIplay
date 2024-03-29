import Foundation

struct Measures: Hashable {
    var trainingStartTime: TimeInterval = 0
    var trainingDuration: TimeInterval = 0
    var trainingAccuracy: Float = 0
    var validationAccuracy: Float = 0
    var trainingLoss: Array<Float>?
}

extension Measures: CustomStringConvertible {
    var description: String {
        "Measures(trainingStartTime: \(trainingStartTime), trainingDuration: \(trainingDuration), trainingAccuracy: \(trainingAccuracy), validationAccuracy: \(validationAccuracy), trainingLoss: \(trainingLoss == nil ? "nil" : "[\(stringOfElements(in: trainingLoss!, count: 16, format: { String(describing: $0) }))]"))"
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

        guard let trainingLossCount = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var trainingLoss: [Float]?
        if trainingLossCount > 0 {
            trainingLoss = [Float]()
            for _ in 0..<trainingLossCount {
                guard let loss = Float(from: data) else { return nil }
                data = data.advanced(by: MemoryLayout<Float>.size)
                trainingLoss!.append(loss)
            }
        }
        
        self.init(trainingStartTime: trainingStartTime, trainingDuration: trainingDuration, trainingAccuracy: trainingAccuracy, validationAccuracy: validationAccuracy, trainingLoss: trainingLoss)
    }
    
    var encode: Data {
        var data = trainingStartTime.encode
        data += trainingDuration.encode
        data += trainingAccuracy.encode
        data += validationAccuracy.encode
        if let trainingLoss = trainingLoss {
            data += trainingLoss.count.encode
            trainingLoss.forEach { data += $0.encode }
        } else {
            data += Int(0).encode
        }
        return data.count.encode + data
    }
}
