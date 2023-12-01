import PlaygroundTester
import Foundation

@objcMembers
final class CustomCoderTests: TestCase {
    internal func testArrayCoder() {
        [
            (input: [1.23, 2.34, 3.45], expect: [1.23, 2.34, 3.45]),
            (input: [1.23, -2.34, 3.45], expect: [1.23, -2.34, 3.45])
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let result = Array<Double>(from: testCase.input.encode)
            AssertEqual(testCase.expect, other: result, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    func testStringCoder() {
        [
            (input: "", expect: ""),
            (input: "ABC", expect: "ABC"),
            (input: "123", expect: "123")
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let result = String(from: testCase.input.encode)
            AssertEqual(testCase.expect, other: result, message: "#\(testCaseIndex + 1) failed")
        }
    }

    internal func testNumericCoder() {
        [
            4711, -4711, Int.min, Int.max
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let input = testCase.encode
            let expect = Int(from: input)
            AssertEqual(expect, other: testCase, message: "#\(testCaseIndex + 1) failed")
        }
        [
            4711.0815, -4711.0815, Float.infinity
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let input = testCase.encode
            let expect = Float(from: input)
            AssertEqual(expect, other: testCase, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testBoolCoder() {
        [
            true, false
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let input = testCase.encode
            let expect = Bool(from: input)
            AssertEqual(expect, other: testCase, message: "#\(testCaseIndex + 1) failed")
        }
    }
}
