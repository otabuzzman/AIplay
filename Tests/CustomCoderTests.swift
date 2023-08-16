import PlaygroundTester
import Foundation

@objcMembers
final class CustomCoderTests: TestCase {
    internal func testStringCoder() {
        [
            (input: "", result: ""),
            (input: "ABC", result: "ABC"),
            (input: "123", result: "123")
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let x = String(from: testCase.input.encode)
            AssertEqual(testCase.result, other: x, message: "#\(testCaseIndex + 1) failed")
        }
    }
    
    internal func testNumericCoder() {
        [
            4711, -4711, Int.min, Int.max
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let input = testCase.encode
            let result = Int(from: input) ?? 0
            AssertEqual(result, other: testCase, message: "#\(testCaseIndex + 1) failed")
        }
        [
            4711.0815, -4711.0815, Float.infinity
        ].enumerated().forEach { (testCaseIndex, testCase) in
            let input = testCase.encode
            let result = Float(from: input) ?? 0
            AssertEqual(result, other: testCase, message: "#\(testCaseIndex + 1) failed")
        }
    }
}
