import PlaygroundTester
import Foundation

@objcMembers
final class MatrixTests: TestCase {
    internal func testMatrixInit() {
        [
            (A: Matrix<Int>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), result: Matrix<Int>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])),
            (A: Matrix<Int>(rows: 2, columns: 3), result: Matrix<Int>(rows: 2, columns: 3, entries: [0, 0, 0, 0, 0, 0])),
            (A: Matrix<Int>(rows: 3), result: Matrix<Int>(rows: 3, columns: 1, entries: [0, 0, 0])),
            (A: Matrix<Int>(columns: 3), result: Matrix<Int>(rows: 1, columns: 3, entries: [0, 0, 0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            AssertEqual(testCase.result, other: testCase.A, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixTransform() {
        [
            (A: Matrix<Int>(rows: 2, columns: 3, entries: [1, -2, 3, 4, -5, 6]), result: Matrix<Int>(rows: 2, columns: 3, entries: [0, 0, 0, 0, 0, 0])),
            (A: Matrix<Int>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), result: Matrix<Int>(rows: 2, columns: 2, entries: [0, 0, 0, 0])),
            (A: Matrix<Int>(rows: 1, columns: 3, entries: [1, -2, 3]), result: Matrix<Int>(rows: 1, columns: 3, entries: [0, 0, 0])),
            (A: Matrix<Int>(rows: 3, columns: 1, entries: [3, -2, 1]), result: Matrix<Int>(rows: 3, columns: 1, entries: [0, 0, 0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let A = testCase.A.map { entry in
                 entry - entry
            }
            AssertEqual(testCase.result, other: A, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixTranspose() {
        [
            (A: Matrix<Int>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), result: Matrix<Int>(rows: 3, columns: 2, entries: [1, 4, 2, 5, 3, 6])),
            (A: Matrix<Int>(rows: 2, columns: 2, entries: [1, 2, 3, 4]), result: Matrix<Int>(rows: 2, columns: 2, entries: [1, 3, 2, 4])),
            (A: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), result: Matrix<Int>(rows: 3, columns: 1, entries: [1, 2, 3])),
            (A: Matrix<Int>(rows: 3, columns: 1, entries: [3, 2, 1]), result: Matrix<Int>(rows: 1, columns: 3, entries: [3, 2, 1]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.A.T
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixScalarAddition () {
        [
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [3.0, 4.0, 5.0, 6.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [3.0, 4.0, 5.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [3.0, 4.0, 5.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [-1.0, 0.0, 1.0, 2.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, 2.0, -3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [-1.0, 0.0, -5.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [-1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [-3.0, 0.0, 1.0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var B = testCase.s + testCase.A
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
            B = testCase.A + testCase.s
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMatrixAddition() {
        [
            (A: Matrix<Int>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), B: Matrix<Int>(rows: 2, columns: 3, entries: [1, 4, 2, 5, 3, 6]), result: Matrix<Int>(rows: 2, columns: 3, entries: [2, 6, 5, 9, 8, 12])),
            (A: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), B: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), result: Matrix<Int>(rows: 1, columns: 3, entries: [2, 4, 6])),
            (A: Matrix<Int>(rows: 3, columns: 1, entries: [1, 2, 3]), B: Matrix<Int>(rows: 3, columns: 1, entries: [1, -2, 3]), result: Matrix<Int>(rows: 3, columns: 1, entries: [2, 0, 6]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let C = testCase.A + testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testScalarMinusMatrix () {
        [
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 0, -1.0, -2.0, -3.0, -4.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 0, -1.0, -2.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, 0, -1.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, 0, -1.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [-3.0, -4.0, -5.0, -6.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [-1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [-1.0, -4.0, -5.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, 2.0, -3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [-3.0, -4.0, 1.0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.s - testCase.A
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMinusScalar () {
        [
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [-1.0, 0, 1.0, 2.0, 3.0, 4.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [-1.0, 0, 1.0, 2.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [-1.0, 0, 1.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [-1.0, 0, 1.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [3.0, 4.0, 5.0, 6.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, 2.0, -3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [3.0, 4.0, -1.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, -2.0, 3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [3.0, 0.0, 5.0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let B = testCase.A - testCase.s
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMatrixSubtraction() {
        [
            (A: Matrix<Int>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), B: Matrix<Int>(rows: 2, columns: 3, entries: [1, 4, 2, 5, 3, 6]), result: Matrix<Int>(rows: 2, columns: 3, entries: [0, -2, 1, -1, 2, 0])),
            (A: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), B: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), result: Matrix<Int>(rows: 1, columns: 3, entries: [0, 0, 0])),
            (A: Matrix<Int>(rows: 3, columns: 1, entries: [1, 2, 3]), B: Matrix<Int>(rows: 3, columns: 1, entries: [1, -2, 3]), result: Matrix<Int>(rows: 3, columns: 1, entries: [0, 4, 0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let C = testCase.A - testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixScalarMultiplication () {
        [
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [2.0, 4.0, 6.0, 8.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [2.0, 4.0, 6.0])),
            (s: Float(2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, 2.0, 3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [2.0, 4.0, 6.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 3, entries: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), result: Matrix<Float>(rows: 2, columns: 3, entries: [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 2, columns: 2, entries: [1.0, 2.0, 3.0, 4.0]), result: Matrix<Float>(rows: 2, columns: 2, entries: [-2.0, -4.0, -6.0, -8.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 1, columns: 3, entries: [1.0, -2.0, 3.0]), result: Matrix<Float>(rows: 1, columns: 3, entries: [-2.0, 4.0, -6.0])),
            (s: Float(-2.0), A: Matrix<Float>(rows: 3, columns: 1, entries: [1.0, -2.0, 3.0]), result: Matrix<Float>(rows: 3, columns: 1, entries: [-2.0, 4.0, -6.0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var B = testCase.s * testCase.A
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
            B = testCase.A * testCase.s
            AssertEqual(testCase.result, other: B, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixMatrixMultiplication() {
        [
            (A: Matrix<Int>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), B: Matrix<Int>(rows: 2, columns: 3, entries: [1, 4, 2, 5, 3, 6]), result: Matrix<Int>(rows: 2, columns: 3, entries: [1, 8, 6, 20, 15, 36])),
            (A: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), B: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), result: Matrix<Int>(rows: 1, columns: 3, entries: [1, 4, 9])),
            (A: Matrix<Int>(rows: 3, columns: 1, entries: [1, 2, 3]), B: Matrix<Int>(rows: 3, columns: 1, entries: [1, 2, 3]), result: Matrix<Int>(rows: 3, columns: 1, entries: [1, 4, 9]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var C = testCase.A * testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
            C = testCase.B * testCase.A
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }

    internal func testMatrixDotProduct() {
        [
            (A: Matrix<Int>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), B: Matrix<Int>(rows: 3, columns: 2, entries: [1, 4, 2, 5, 3, 6]), result: Matrix<Int>(rows: 2, columns: 2, entries: [14, 32, 32, 77])),
            (A: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), B: Matrix<Int>(rows: 3, columns: 1, entries: [1, 2, 3]), result: Matrix<Int>(rows: 1, columns: 1, entries: [14])),
            (A: Matrix<Int>(rows: 3, columns: 1, entries: [1, 2, 3]), B: Matrix<Int>(rows: 1, columns: 3, entries: [1, 2, 3]), result: Matrix<Int>(rows: 3, columns: 3, entries: [1, 2, 3, 2, 4, 6, 3, 6, 9]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let C = testCase.A • testCase.B
            AssertEqual(testCase.result, other: C, message: "#\(testCaseIndex + 1) failed")
        }
    }
}