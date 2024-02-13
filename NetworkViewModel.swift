extension NetworkView {
    class NetworkViewModel: ObservableObject {
        var network: Network!
        private(set) var dataset: MNISTViewModel!
        private(set) var epochsWanted: Int!
        private(set) var miniBatchSize: Int!
        
        @Published var progress: Float = 0 // 0...1
        
        init(config: NetworkConfig? = nil) {
            if let config = config {
                _ = setup(config: config)
            } else {
                reset()
            }
            dataset = MNISTViewModel()
        }
        
        func reset() -> Void {
            let model = Bundle.main.url(forResource: "default-model", withExtension: "nndx")!
            network = try! Network(from: Data(contentsOf: model))! // should not fail
            
            _ = setup(config: .default)
        }
        
        func setup(config: NetworkConfig) -> Bool {
            guard
                let network = GenericFactory.create(NetworkFactory(), config)
            else { return false }
            self.network = network
            
            epochsWanted = config.epochsWanted
            miniBatchSize = config.miniBatchSize
            
            return true
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
            network.query(for: image.toInput()).entries
        }
        
        func train() async -> Measures {
            let measures = Measures()
            measures.trainingStartTime = Date.timeIntervalSinceReferenceDate
            defer {
                measures.trainingDuration = Date.timeIntervalSinceReferenceDate - measures.trainingStartTime
            }
            
            let batches = dataset.count(in: .train) / miniBatchSize
            measures.trainingLoss = .init(repeating: 0, count: batches)
            
            for batch in 0..<batches {
                let start = batch * miniBatchSize
                let end = start + miniBatchSize
                let cost = await train(batch: start..<end)
                measures.trainingLoss?[batch] = cost
                
                let current = Float(batch) / Float(batches)
                advanceProgress(current)
                
                do {
                    try Task.checkCancellation()
                } catch { return measures }
            }
            
            measures.trainingAccuracy = await query(subset: .train)
            measures.validationAccuracy = await query(subset: .test)
            
            if _isDebugAssertConfiguration() {
                let _ = print(measures)
            }
            
            return measures
        }
        
        func train(batch range: Range<Int>) async -> Float {
            let (images, labels) = dataset.fetch(range, from: .train)
            let cost = await network.train(for: images.toMatrix(), with: labels.toMatrix())
            return cost
        }
        
        func train(image index: Int) async -> Float {
            let (image, label) = dataset.fetch(index, from: .train)
            let loss = await network.train(for: image.toInput(), with: label.toTarget())
            return loss
        }
        
        private func advanceProgress(_ current: Float) -> Void {
            if current - progress > 0.01 {
                progress += current
            }
        }
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
