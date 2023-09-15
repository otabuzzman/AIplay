import GameplayKit

// https://stackoverflow.com/a/49471411
class GaussianDistribution {
    private let randomSource: GKRandomSource
    private let mean: Float
    private let deviation: Float
    
    init?(randomSource: GKRandomSource, mean: Float, deviation: Float) {
        guard
            deviation >= 0
        else { return nil }
        self.randomSource = randomSource
        self.mean = mean
        self.deviation = deviation
    }
    
    func nextFloat() -> Float {
        guard
            deviation > 0
        else { return mean }
        // https://mathworld.wolfram.com/Box-MullerTransformation.html
        let x1 = randomSource.nextUniform() // 0...1
        let x2 = randomSource.nextUniform() // 0...1
        let z1 = sqrt(-2 * log(x1)) * cos(2 * .pi * x2)
        return z1 * deviation + mean 
    }
}
