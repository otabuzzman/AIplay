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
    var encode: Data {
        withUnsafeBytes(of: self.bigEndian) { Data($0) }
    }
    
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

extension String: CustomCoder {
    var encode: Data {
        Data(self.utf8)
    }

    init?(from: Data) {
        self.init(data: from, encoding: .utf8)
    }
}

extension Bool: CustomCoder {
    var encode: Data {
        Data(UInt8(self ? 1 : 0).encode)
    }
    
    init?(from: Data) {
        guard let value = UInt8(from: from) else { return nil }
        self.init(value == 1)
    }
}
