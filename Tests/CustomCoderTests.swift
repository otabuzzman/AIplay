import PlaygroundTester
import Foundation

@objcMembers
final class CustomCoderTests: TestCase {
    func testStringCoder() {
        [
            (input: "", expect: ""),
            (input: "ABC", expect: "ABC"),
            (input: "123", expect: "123"),
            (input: "123🙂", expect: "123🙂")
        ].enumerated().forEach { (testCaseIndex, testCase) in
            var data = testCase.input.encode
            let size = Int(from: data)!
            data = data.advanced(by: MemoryLayout<Int>.size)
            let result = String(from: data, size: size)
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
