import Foundation

typealias CustomCoder = CustomEncoder & CustomDecoder

protocol CustomEncoder {
    var encode: Data { get }
}

protocol CustomDecoder {
    init?(from: Data)
}

protocol CustomNumericCoder: CustomCoder where Self: ExpressibleByIntegerLiteral {
    var bigEndian: Self { get }
}

// https://stackoverflow.com/a/38024025
extension CustomNumericCoder {
    init?(from: Data) {
        guard
            from.count >= MemoryLayout<Self>.size
        else { return nil }
        var value: Self = 0 // ExpressibleByIntegerLiteral
        _ = withUnsafeMutableBytes(of: &value) {
            from.copyBytes(to: $0, count: MemoryLayout<Self>.size)
        }
        self = value.bigEndian
    }
    
    var encode: Data {
        withUnsafeBytes(of: self.bigEndian) { Data($0) }
    }
}

extension Int: CustomNumericCoder { }
extension Int32: CustomNumericCoder { }
extension Int16: CustomNumericCoder { }
extension Int8: CustomNumericCoder { }
extension UInt: CustomNumericCoder { }
extension UInt32: CustomNumericCoder { }
extension UInt16: CustomNumericCoder { }
extension UInt8: CustomNumericCoder { }

extension Float: CustomNumericCoder {
    var bigEndian: Self { Self(bitPattern: self.bitPattern.bigEndian) }    
}

extension Double: CustomNumericCoder {
    var bigEndian: Self { Self(bitPattern: self.bitPattern.bigEndian) }    
}

protocol CustomStringDecoder {
    init?(from: Data, bytes: Int)
}

typealias CustomStringCoder = CustomEncoder & CustomStringDecoder

extension String: CustomStringCoder {
    init?(from: Data, bytes: Int) {
        self.init(data: from[..<bytes], encoding: .utf8)
    }

    var encode: Data {
        self.utf8.count.encode + self.utf8
    }
}

extension Bool: CustomCoder {
    init?(from: Data) {
        guard let value = UInt8(from: from) else { return nil }
        self.init(value == 1)
    }

    var encode: Data {
        Data(UInt8(self ? 1 : 0).encode)
    }
}
