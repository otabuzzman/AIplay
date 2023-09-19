import Foundation

typealias CustomCoder = CustomEncoder & CustomDecoder

protocol CustomEncoder {
    var encode: Data { get }
}

extension CustomEncoder {
    var encode: Data {
        withUnsafeBytes(of: self) { Data($0) }
    }
}

protocol CustomDecoder {
    init?(from: Data)
}

// https://stackoverflow.com/a/38024025
extension CustomDecoder where Self: ExpressibleByIntegerLiteral {
    init?(from: Data) {
        guard
            from.count >= MemoryLayout<Self>.size
        else { return nil }
        var value: Self = 0 // ExpressibleByIntegerLiteral
        _ = withUnsafeMutableBytes(of: &value) {
            from.copyBytes(to: $0, count: MemoryLayout<Self>.size)
        }
        self = value
    }
}

protocol CustomNumericCoder: CustomCoder {
    var bigEndian: Self { get }
}

extension Int: CustomNumericCoder { }
extension Int32: CustomNumericCoder { }
extension UInt: CustomNumericCoder { }
extension UInt32: CustomNumericCoder { }

extension Float: CustomNumericCoder {
    var bigEndian: Self { Self(bitPattern: self.bitPattern.bigEndian) }    
}

extension Double: CustomNumericCoder {
    var bigEndian: Self { Self(bitPattern: self.bitPattern.bigEndian) }    
}

extension String: CustomEncoder {
    var encode: Data {
        Data( self.utf8 )
    }
}

extension String: CustomDecoder {
    init?(from: Data) {
        self.init(data: from, encoding: .utf8)
    }
}
