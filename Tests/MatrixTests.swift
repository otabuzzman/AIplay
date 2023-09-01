import PlaygroundTester
import Foundation

@objcMembers
final class MatrixTests: TestCase {
    internal func testMatrixInit() {
        [
            (A: Matrix<Float>(), result: Matrix<Float>(rows: 1, columns: 1, entries: [0])),
            (A: Matrix<Float>(rows: 2), result: Matrix<Float>(rows: 2, columns: 1, entries: [0, 0])),
            (A: Matrix<Float>(columns: 3), result: Matrix<Float>(rows: 1, columns: 3, entries: [0, 0, 0])),
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])),
        ].enumerated().forEach { (testCaseIndex, testCase) in
            AssertEqual(testCase.result, other: testCase.A, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixTransform() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [0, 0, 0, 0, 0, 0])),
            (A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), result: Matrix<Float>(rows: 2, columns: 2, entries: [0, 0, 0, 0])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, -2, 3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [0, 0, 0])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [3, -2, 1]), result: Matrix<Float>(rows: 3, columns: 1, entries: [0, 0, 0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let A = testCase.A.map { entry in
                 entry - entry
            }
            AssertEqual(testCase.result, other: A, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixTranspose() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), result: Matrix<Float>(rows: 3, columns: 2, entries: [1, -4, -2, 5, 3, -6])),
            (A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), result: Matrix<Float>(rows: 2, columns: 2, entries: [-1, -3, 2, 4])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, -2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [1, -2, 3])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [3, -2, 1]), result: Matrix<Float>(rows: 1, columns: 3, entries: [3, -2, 1]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.A.T
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixScalarAddition() {
        [
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), result: Matrix<Float>(rows: 3, columns: 2, entries: [3, 0, 5, -2, 7, -4])),
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), result: Matrix<Float>(rows: 2, columns: 2, entries: [1, 4, -1, 6])),
            (s: Float(-2), A: Matrix<Float>(rows: 1, columns: 3, entries: [1, -2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [-1, -4, 1])),
            (s: Float(-2), A: Matrix<Float>(rows: 3, columns: 1, entries: [3, -2, 1]), result: Matrix<Float>(rows: 1, columns: 3, entries: [1, -4, -1]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var B = testCase.s + testCase.A
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
            B = testCase.A + testCase.s
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMatrixAddition() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), B: Matrix<Float>(rows: 2, columns: 3, entries: [1, 4, 2, 5, 3, 6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [2, 4, 6, 8, 10, 12])),
            (A: Matrix<Float>(rows: 2, columns: 2, entries: [1, 2, 3, 4]), B: Matrix<Float>(rows: 2, columns: 2, entries: [1, 2, 3, 4]), result: Matrix<Float>(rows: 2, columns: 3, entries: [2, 4, 6, 8])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, 3]), B: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, 3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [2, 4, 6])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [1, 2, 3]), B: Matrix<Float>(rows: 3, columns: 1, entries: [1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [2, 4, 6]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var C = testCase.A + testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
            C = testCase.B + testCase.A
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testScalarMatrixSubtraction() {
        [
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [1, 0, -1, -2, -3, -4])),
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 2, entries: [1, 2, 3, 4]), result: Matrix<Float>(rows: 2, columns: 2, entries: [1, 0, -1, -2])),
            (s: Float(-2), A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [-3, -4, 1])),
            (s: Float(-2), A: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [-1, -4, -5]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.s - testCase.A
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixScalarSubtraction() {
        [
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [-1, 0, 1, 2, 3, 4])),
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 2, entries: [1, 2, 3, 4]), result: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 0, 1, 2])),
            (s: Float(-2), A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [3, 4, -1])),
            (s: Float(-2), A: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [1, 4, 5]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.A - testCase.s
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMatrixSubtraction() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), B: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [0, 0, 0, 0, 0, 0])),
            (A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), B: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), result: Matrix<Float>(rows: 2, columns: 3, entries: [0, 0, 0, 0])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), B: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [0, 0, 0])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), B: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [0, 0, 0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let C = testCase.A - testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixScalarMultiplication() {
        [
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [2, -4, 6, -8, 10, -12])),
            (s: Float(2), A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), result: Matrix<Float>(rows: 2, columns: 2, entries: [-2, 4, -6, 8])),
            (s: Float(-2), A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [-2, -4, 6])),
            (s: Float(-2), A: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [2, -4, -6]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var B = testCase.s * testCase.A
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
            B = testCase.A * testCase.s
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMatrixMultiplication() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), B: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), result: Matrix<Float>(rows: 2, columns: 3, entries: [1, 4, 9, 16, 25, 36])),
            (A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), B: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), result: Matrix<Float>(rows: 2, columns: 3, entries: [1, 4, 9, 16])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), B: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [1, 4, 9])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), B: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [1, 4, 9]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var C = testCase.A * testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
            C = testCase.B * testCase.A
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testScalarMatrixDivision() {
        [
            (s: Float(6), A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3]), result: Matrix<Float>(rows: 1, columns: 3, entries: [6, 3, -2])),
            (s: Float(-6), A: Matrix<Float>(rows: 3, columns: 1, entries: [-1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 1, entries: [6, -3, -2]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.s / testCase.A
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixScalarDivision() {
        [
            (s: Float(6), A: Matrix<Float>(rows: 1, columns: 3, entries: [6, 12, -18]), result: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, -3])),
            (s: Float(-6), A: Matrix<Float>(rows: 3, columns: 1, entries: [-6, 12, 18]), result: Matrix<Float>(rows: 3, columns: 1, entries: [1, -2, -3]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.A / testCase.s
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMatrixDivision() {
        [
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [6, 12, -18]), B: Matrix<Float>(rows: 1, columns: 3, entries: [-2, 4, 6]), result: Matrix<Float>(rows: 1, columns: 3, entries: [-3, 3, -3])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [-6, 12, 18]), B: Matrix<Float>(rows: 3, columns: 1, entries: [2, 4, -6]), result: Matrix<Float>(rows: 3, columns: 1, entries: [-3, 3, -3]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let C = testCase.A / testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixDotProduct() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), B: Matrix<Float>(rows: 3, columns: 2, entries: [1, 4, 2, 5, 3, 6]), result: Matrix<Float>(rows: 2, columns: 2, entries: [14, 32, 32, 77])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, 3]), B: Matrix<Float>(rows: 3, columns: 1, entries: [1, 2, 3]), result: Matrix<Float>(rows: 1, columns: 1, entries: [14])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [1, 2, 3]), B: Matrix<Float>(rows: 1, columns: 3, entries: [1, 2, 3]), result: Matrix<Float>(rows: 3, columns: 3, entries: [1, 2, 3, 2, 4, 6, 3, 6, 9]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let C = testCase.A • testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
}
