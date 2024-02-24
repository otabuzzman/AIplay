import SwiftUI

extension NetworkView {
    class NetworkViewModel: ObservableObject {
        var nnxd: NNXD
        private(set) var dataset = MNISTViewModel()
        
        @Published var progress: Float = 0 // 0...1
        
        init() {
            nnxd = Bundle.nnxd(name: "default-model")!
            setNetworkConfig(.default)
        }
        
        func query(subset: MNISTSubset.Purpose = .test) async -> Float {
            let samples = dataset.count(in: subset)
            var results = [Int](repeating: 0, count: samples)
            
            for sample in 0..<samples {
                let (image, label) = dataset.fetch(sample, from: subset)
                let result = query(image: image).maxElementIndex()
                results[sample] = result ?? -1 == label ? 1 : 0
                
                let current = Float(sample) / Float(samples)
                advanceProgress(current)
                
                do {
                    try Task.checkCancellation()
                } catch { return 0 }
            }
            
            let accuracy = Float(results.reduce(0, +)) / Float(samples)
            return accuracy
        }
        
        func query(image index: Int) -> [Float] {
            let (image, _) = dataset.fetch(index, from: .test)
            return query(image: image)
        }
        
        func query(image: MNISTImage) -> [Float] {
            nnxd.network.query(for: image.toInput()).entries
        }
        
        func train() async -> Void {
            dataset.shuffle()
            
            var measures = Measures()
            measures.trainingStartTime = Date.timeIntervalSinceReferenceDate
            
            let batches = dataset.count(in: .train) / nnxd.miniBatchSize
            measures.trainingLoss = .init(repeating: 0, count: 1000)
            
            let measureInterval = batches > 1000 ? batches / 1000 : 1
            
            for batch in 0..<batches {
                let start = batch * nnxd.miniBatchSize
                let end = start + nnxd.miniBatchSize
                
                var cost: Float
                if nnxd.miniBatchSize == 1 {
                    cost = await train(image: batch)
                } else {
                    cost = await train(batch: start..<end)
                }
                
                if batch % measureInterval == 0 {
                    measures.trainingLoss?[batch / measureInterval] = cost
                }
                
                let current = Float(batch) / Float(batches)
                advanceProgress(current)
                
                do {
                    try Task.checkCancellation()
                } catch { return }
            }
            
            measures.trainingAccuracy = await query(subset: .train)
            measures.validationAccuracy = await query(subset: .test)
            
            measures.trainingDuration = Date.timeIntervalSinceReferenceDate - measures.trainingStartTime
            nnxd.measures.append(measures)
            
            if _isDebugAssertConfiguration() {
                let _ = print(measures)
            }
        }
        
        func train(batch range: Range<Int>) async -> Float {
            let (images, labels) = dataset.fetch(range, from: .train)
            let cost = await nnxd.network.train(for: images.toMatrix(), with: labels.toMatrix())
            return cost
        }
        
        func train(image index: Int) async -> Float {
            let (image, label) = dataset.fetch(index, from: .train)
            let loss = await nnxd.network.train(for: image.toInput(), with: label.toTarget())
            return loss
        }
        
        private func advanceProgress(_ current: Float) -> Void {
            if progress > current {
                progress = 0
            }
            if current - progress > 0.01 {
                progress = current
            }
        }
    }
}

extension Bundle {
    static func nnxd(name: String) -> NNXD? {
        guard
            let model = Self.main.url(forResource: name, withExtension: "nnxd"),
            let data = try? Data(contentsOf: model),
            let nnxd = NNXD(from: data)
        else { return nil }
        return nnxd
    }
}

extension Array {
    func toMatrix() -> [Matrix<Float>] where Element == MNISTImage {
        self.map { $0.toInput() }
    }
    
    func toMatrix() -> [Matrix<Float>] where Element == MNISTLabel {
        self.map { $0.toTarget() }
    }
}

extension MNISTImage {
    func toInput() -> Matrix<Float> {
        Matrix<Float>(
            rows: self.count, columns: 1,
            entries: self.map { (Float($0) / 255.0 * 0.99) + 0.01 }) // MYONN, p. 151 ff.
    }
}

extension MNISTLabel {
    func toTarget() -> Matrix<Float> {
        var target = Matrix<Float>(rows: 10, columns: 1)
            .map { _ in 0.01 }
        target[Int(self), 0] = 0.99
        return target
    }
}

extension Array where Element: Comparable {
    func maxElementValue() -> Element? {
        self.max(by: { $0 < $1 })
    }
    
    func maxElementIndex() -> Int? {
        self.indices.max(by: { self[$0] < self[$1] })
    }
}
