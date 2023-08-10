import Foundation

protocol CustomEncodable {
    func encode() throws -> Data
}

protocol CustumDecodable {
    init?(from: Data)
}

typealias CustomCodable = CustomEncodable & CustumDecodable

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

protocol CustomNumericCoder: CustomEncoder, CustomDecoder {
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

extension Data {
    func description() -> String { // var description set by Foundation
        var text = String(format: "Data(bytes: [", count)
        let list: (Int, Int) -> String = { startIndex, count in
            var text: String = ""
            let endIndex = startIndex + count - 1
            for index in startIndex...endIndex {
                text += String(format: "0x%02X", self[index])
                if index == endIndex { break }
                text += ", "
            }
            return text
        }
        if count > 10 {
            text += list(0, 8) + ", ... " + list(count - 2, 2)
        } else {
            text += list(0, count)
        }
        return text + String(format: "], count: %d)", count)
    }
}
