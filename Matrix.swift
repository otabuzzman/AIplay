import Accelerate
import Foundation

struct Matrix<Entry: Numeric> {
    private(set) var rows: Int
    private(set) var columns: Int
    private(set) var entries: [Entry]
    
    var T: Self {
        var transpose = Self(rows: columns, columns: rows)
        for r in 0..<rows {
            for c in 0..<columns {
                transpose[c, r] = self[r, c]
            }
        }
        return transpose
    }
    
    init(rows: Int, columns: Int, entries: [Entry]) {
        assert(rows > 0 && columns > 0 && rows * columns == entries.count, "wrong dimensions")
        self.rows = rows
        self.columns = columns
        self.entries = entries
    }
    
    init(rows: Int = 1, columns: Int = 1) {
        assert(rows > 0 && columns > 0, "wrong dimensions")
        let entries = Array<Entry>(repeating: .zero, count: rows * columns)
        self.init(rows: rows, columns: columns, entries: entries)
    }
    
    subscript(row: Int, column: Int) -> Entry {
        get {
            assert(indexIsValid(row: row, column: column), "index out of bounds")
            return entries[(row * columns) + column]
        }
        
        set(value) {
            assert(indexIsValid(row: row, column: column), "index out of bounds")
            entries[(row * columns) + column] = value
        }
    }
    
    func map(_ transform: (Entry) throws -> Entry) rethrows -> Self {
        var entries = self.entries
        for e in 0..<entries.count {
            entries[e] = try transform(entries[e])
        }
        return Self(rows: rows, columns: columns, entries: entries)
    }
    
    private func indexIsValid(row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
}

infix operator •: MultiplicationPrecedence
extension Matrix {
    static func +(lhs: Entry, rhs: Self) -> Self {
        rhs.map { lhs + $0 }
    }
    
    static func +(lhs: Self, rhs: Entry) -> Self {
        lhs.map { $0 + rhs }
    }
    
    static func +(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        let entries = lhs.entries.indices.map { lhs.entries[$0] + rhs.entries[$0] }
        return Self(rows: lhs.rows, columns: lhs.columns, entries: entries)
    }
    
    static func +=(lhs: inout Self, rhs: Self) {
        lhs = lhs + rhs
    }
    
    static func -(lhs: Entry, rhs: Self) -> Self {
        rhs.map { lhs - $0 }
    }
    
    static func -(lhs: Self, rhs: Entry) -> Self {
        lhs.map { $0 - rhs }
    }
    
    static func -(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        let entries = lhs.entries.indices.map { lhs.entries[$0] - rhs.entries[$0] }
        return Self(rows: lhs.rows, columns: lhs.columns, entries: entries)
    }
    
    static func *(lhs: Entry, rhs: Self) -> Self {
        rhs.map { lhs * $0 }
    }
    
    static func *(lhs: Self, rhs: Entry) -> Self {
        lhs.map { $0 * rhs }
    }
    
    static func *(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        let entries = lhs.entries.indices.map { lhs.entries[$0] * rhs.entries[$0] }
        return Self(rows: lhs.rows, columns: lhs.columns, entries: entries)
    }
    
    static func •(lhs: Self, rhs: Self) -> Self {
        assert(lhs.columns == rhs.rows, "LHS and RHS dimensions not matching")
        var result = Self(rows: lhs.rows, columns: rhs.columns)
        for r in 0..<lhs.rows {
            for c in 0..<rhs.columns {
                for e in 0..<rhs.rows { // or lhs.columns
                    result[r, c] += lhs[r, e] * rhs[e, c]
                }
            }
        }
        return result
    }
}

// https://forums.swift.org/t/whats-the-most-general-type-operator-can-be-used-on/51068
extension Matrix where Entry == Int {
    static func /(lhs: Entry, rhs: Self) -> Self {
        rhs.map { lhs / $0 }
    }
    
    static func *(lhs: Self, rhs: Entry) -> Self {
        lhs.map { $0 / rhs }
    }
    
    static func *(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        let entries = lhs.entries.indices.map { lhs.entries[$0] / rhs.entries[$0] }
        return Self(rows: lhs.rows, columns: lhs.columns, entries: entries)
    }
}

extension Matrix where Entry == Float {
    static func +(lhs: Entry, rhs: Self) -> Self {
        Self(rows: rhs.rows, columns: rhs.columns, entries: vDSP.add(lhs, rhs.entries))
    }
    
    static func +(lhs: Self, rhs: Entry) -> Self {
        Self(rows: lhs.rows, columns: lhs.columns, entries: vDSP.add(rhs, lhs.entries))
    }
    
    static func +(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        return Self(rows: lhs.rows, columns: lhs.columns, entries: vDSP.add(lhs.entries, rhs.entries))
    }
    
    static func +=(lhs: inout Self, rhs: Self) {
        lhs = lhs + rhs
    }
    
    static prefix func -(M: Self) -> Self {
        M.map { -$0 }
    }
    
    static func -(lhs: Entry, rhs: Self) -> Self {
        -rhs + lhs
    }
    
    static func -(lhs: Self, rhs: Entry) -> Self {
        lhs + -rhs
    }
    
    static func -(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        return Self(rows: lhs.rows, columns: lhs.columns, entries: vDSP.subtract(lhs.entries, rhs.entries))
    }
    
    static func *(lhs: Entry, rhs: Self) -> Self {
        Self(rows: rhs.rows, columns: rhs.columns, entries: vDSP.multiply(lhs, rhs.entries))
    }
    
    static func *(lhs: Self, rhs: Entry) -> Self {
        Self(rows: lhs.rows, columns: lhs.columns, entries: vDSP.multiply(rhs, lhs.entries))
    }

    static func *(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        return Self(rows: lhs.rows, columns: lhs.columns, entries: vDSP.multiply(lhs.entries, rhs.entries))
    }
    
    static func /(lhs: Entry, rhs: Self) -> Self {
        Self(rows: rhs.rows, columns: rhs.columns, entries: vDSP.divide(lhs, rhs.entries))
    }
    
    static func /(lhs: Self, rhs: Entry) -> Self {
        Self(rows: lhs.rows, columns: lhs.columns, entries: vDSP.divide(rhs, lhs.entries))
    }

    static func /(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "LHS and RHS dimensions not matching")
        return Self(rows: lhs.rows, columns: lhs.columns, entries: vDSP.divide(lhs.entries, rhs.entries))
    }
    
    static func •(lhs: Self, rhs: Self) -> Self {
        assert(lhs.columns == rhs.rows, "LHS and RHS dimensions not matching")
        var result = Self(rows: lhs.rows, columns: rhs.columns)
        let stride = vDSP_Stride(1)
        vDSP_mmul(
            lhs.entries, stride,
            rhs.entries, stride,
            &result.entries, stride,
            vDSP_Length(lhs.rows),
            vDSP_Length(rhs.columns),
            vDSP_Length(lhs.columns))
        return result
    }
}

extension Matrix: Equatable {
    static func ==(lhs: Self, rhs: Self) -> Bool {
        if lhs.rows != rhs.rows { return false }
        if lhs.columns != rhs.columns { return false }
        if lhs.entries != rhs.entries { return false }
        return true
    }
    
    static func !=(lhs: Self, rhs: Self) -> Bool {
        return !(lhs == rhs)
    }
}

extension Matrix: CustomStringConvertible {
    var description: String {
        "Matrix(rows: \(rows), columns: \(columns), entries: [\(stringOfElements(in: entries, count: 10, format: { String(describing: $0) }))]"
    }
}

extension Matrix: CustomCoder where Entry: CustomNumericCoder {
    init?(from: Data) {
        var data = from
        
        guard Int(from: data)?.bigEndian != nil else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let rows = Int(from: data)?.bigEndian else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let columns = Int(from: data)?.bigEndian else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let entriesCount = Int(from: data)?.bigEndian else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        var entries = [Entry]()
        for _ in 0..<entriesCount {
            guard let entry = Entry(from: data)?.bigEndian else { return nil }
            entries.append(entry) 
            data = data.advanced(by: MemoryLayout<Entry>.size)
        }
        
        self.init(rows: rows, columns: columns, entries: entries)
    }
    
    var encode: Data {
        var data = rows.bigEndian.encode
        data += columns.bigEndian.encode
        data += entries.count.bigEndian.encode
        entries.forEach { data += $0.bigEndian.encode }
        return data.count.bigEndian.encode + data
    }
}
