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

extension String: CustomEncoder {
    var encode: Data {
        Data(self.utf8)
    }
}

extension String: CustomDecoder {
    init?(from: Data) {
        self.init(data: from, encoding: .utf8)
    }
}

extension Bool: CustomEncoder {
    var encode: Data {
        Data(UInt8(self ? 1 : 0).encode)
    }
}

extension Bool: CustomDecoder {
    init?(from: Data) {
        guard let value = UInt8(from: from) else { return nil }
        self.init(value == 1)
    }
}

extension Array: CustomEncoder where Element: CustomEncoder {
    var encode: Data {
            var data = self.count.encode
            self.forEach { data += $0.encode }
            return data
        }
}

extension Array: CustomDecoder where Element: CustomDecoder {
    init?(from: Data) {
        guard let count = Int(from: from) else { return nil }
        var data = from.advanced(by: MemoryLayout<Int>.size)
        var array = [Element]()
        for _ in 0..<count {
            guard let value = Element(from: data) else { return nil }
            data = data.advanced(by: MemoryLayout<Element>.size)
            array.append(value)
        }
        self.init(array)
    }
}
