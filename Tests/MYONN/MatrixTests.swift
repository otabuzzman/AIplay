import PlaygroundTester
import Foundation

@objcMembers
final class MatrixTests: TestCase {
    internal func testMatrixInit() {
        [
            (A: Matrix<Float>(), expect: Matrix<Float>(rows: 1, columns: 1, entries: [0])),
            (A: Matrix<Float>(rows: 2), expect: Matrix<Float>(rows: 2, columns: 1, entries: [0, 0])),
            (A: Matrix<Float>(columns: 3), expect: Matrix<Float>(rows: 1, columns: 3, entries: [0, 0, 0])),
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6]), expect: Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])),
        ].enumerated().forEach { (testCaseIndex, testCase) in
            AssertEqual(testCase.expect, other: testCase.A, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixTransform() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), expect: Matrix<Float>(rows: 2, columns: 3, entries: [0, 0, 0, 0, 0, 0])),
            (A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), expect: Matrix<Float>(rows: 2, columns: 2, entries: [0, 0, 0, 0])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, -2, 3]), expect: Matrix<Float>(rows: 1, columns: 3, entries: [0, 0, 0])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [3, -2, 1]), expect: Matrix<Float>(rows: 3, columns: 1, entries: [0, 0, 0]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let result = testCase.A.map { entry in
                 entry - entry
            }
            AssertEqual(testCase.expect, other: result, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testMatrixTranspose() {
        [
            (A: Matrix<Float>(rows: 2, columns: 3, entries: [1, -2, 3, -4, 5, -6]), expect: Matrix<Float>(rows: 3, columns: 2, entries: [1, -4, -2, 5, 3, -6])),
            (A: Matrix<Float>(rows: 2, columns: 2, entries: [-1, 2, -3, 4]), expect: Matrix<Float>(rows: 2, columns: 2, entries: [-1, -3, 2, 4])),
            (A: Matrix<Float>(rows: 1, columns: 3, entries: [1, -2, 3]), expect: Matrix<Float>(rows: 3, columns: 1, entries: [1, -2, 3])),
            (A: Matrix<Float>(rows: 3, columns: 1, entries: [3, -2, 1]), expect: Matrix<Float>(rows: 1, columns: 3, entries: [3, -2, 1]))
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let result = testCase.A.T
            AssertEqual(testCase.expect, other: result, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testScalarMatrixAddition() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let s: Float = 1
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [2, 3, 4, 5, 6, 7])
        
        let result = s + A
        
        AssertEqual(expect, other: result, message: "s + A failed")
    }
    
    internal func testMatrixScalarAddition() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let s: Float = 1
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [2, 3, 4, 5, 6, 7])
        
        let result = A + s
        
        AssertEqual(expect, other: result, message: "A + s failed")
    }
    
    internal func testMatrixMatrixAddition() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let B = Matrix<Float>(rows: 2, columns: 3, entries: [6, 5, 4, 3, 2, 1])
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [7, 7, 7, 7, 7, 7])
        
        var result = A + B
        
        AssertEqual(expect, other: result, message: "A + B failed")
        
        result = A
        result += B
        
        AssertEqual(expect, other: result, message: "A + B failed")
    }
    
    internal func testScalarMatrixSubtraction() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let s: Float = 1
        let R = Matrix<Float>(rows: 2, columns: 3, entries: [0, -1, -2, -3, -4, -5])
        
        let result = s - A
        
        AssertEqual(result, other: R, message: "s - A failed")
    }
    
    internal func testMatrixScalarSubtraction() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let s: Float = 1
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [0, 1, 2, 3, 4, 5])
        
        let result = A - s
        
        AssertEqual(expect, other: result, message: "A - s failed")
    }
    
    internal func testMatrixMatrixSubtraction() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let B = Matrix<Float>(rows: 2, columns: 3, entries: [6, 5, 4, 3, 2, 1])
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [-5, -3, -1, 1, 3, 5])
        
        let result = A - B
        
        AssertEqual(expect, other: result, message: "A - B failed")
    }
    
    internal func testScalarMatrixMultiplication() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let s: Float = 2
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [2, 4, 6, 8, 10, 12])
        
        let result = s * A
        
        AssertEqual(expect, other: result, message: "s * A failed")
    }
    
    internal func testMatrixScalarMultiplication() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let s: Float = 2
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [2, 4, 6, 8, 10, 12])
        
        let result = A * s
        
        AssertEqual(expect, other: result, message: "A * s failed")
    }
    
    internal func testMatrixMatrixMultiplication() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let B = Matrix<Float>(rows: 2, columns: 3, entries: [6, 5, 4, 3, 2, 1])
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [6, 10, 12, 12, 10, 6])
        
        let result = A * B
        
        AssertEqual(expect, other: result, message: "A * B failed")
    }
    
    internal func testScalarMatrixDivision() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let s: Float = 60
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [60, 30, 20, 15, 12, 10])
        
        let result = s / A
        
        AssertEqual(expect, other: result, message: "s / A failed")
    }
    
    internal func testMatrixScalarDivision() {
        let C = Matrix<Float>(rows: 2, columns: 3, entries: [10, 20, 30, 40, 50, 60])
        let s: Float = 2
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [5, 10, 15, 20, 25, 30])
        
        let result = C / s
        
        AssertEqual(expect, other: result, message: "C / s failed")
    }
    
    internal func testMatrixMatrixDivision() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let C = Matrix<Float>(rows: 2, columns: 3, entries: [10, 20, 30, 40, 50, 60])
        let expect = Matrix<Float>(rows: 2, columns: 3, entries: [10, 10, 10, 10, 10, 10])
        
        let result = C / A
        
        AssertEqual(expect, other: result, message: "C / A failed")
    }
    
    internal func testMatrixDotProduct() {
        let A = Matrix<Float>(rows: 2, columns: 3, entries: [1, 2, 3, 4, 5, 6])
        let B = Matrix<Float>(rows: 3, columns: 2, entries: [1, 4, 2, 5, 3, 6])
        let expect = Matrix<Float>(rows: 2, columns: 2, entries: [14, 32, 32, 77])
        
        let result = A • B
        
        AssertEqual(expect, other: result, message: "A • B failed")
    }
}
