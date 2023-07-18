struct Matrix<Entry: Numeric> {
    let rows: Int
    let columns: Int
    
    private var entries: [Entry]
    
    var T: Matrix {
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
        self.init(rows: rows, columns: columns, entries: Array<Entry>(repeating: .zero, count: rows * columns))
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
        var result = Self(rows: rhs.rows, columns: rhs.columns)
        for r in 0..<rhs.rows {
            for c in 0..<rhs.columns {
                result[r, c] = lhs + rhs[r, c]
            }
        }
        return result
    }
    
    static func +(lhs: Self, rhs: Entry) -> Self {
        var result = Self(rows: lhs.rows, columns: lhs.columns)
        for r in 0..<lhs.rows {
            for c in 0..<lhs.columns {
                result[r, c] = lhs[r, c] + rhs
            }
        }
        return result
    }
    
    static func +(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "dimensions of LHS and RHS not matching")
        var result = Self(rows: lhs.rows, columns: lhs.columns)
        for r in 0..<lhs.rows {
            for c in 0..<lhs.columns {
                result[r, c] = lhs[r, c] + rhs[r, c]
            }
        }
        return result
    }
    
    static func -(lhs: Entry, rhs: Self) -> Self {
        var result = Self(rows: rhs.rows, columns: rhs.columns)
        for r in 0..<rhs.rows {
            for c in 0..<rhs.columns {
                result[r, c] = lhs - rhs[r, c]
            }
        }
        return result
    }
    
    static func -(lhs: Self, rhs: Entry) -> Self {
        var result = Self(rows: lhs.rows, columns: lhs.columns)
        for r in 0..<lhs.rows {
            for c in 0..<lhs.columns {
                result[r, c] = lhs[r, c] - rhs
            }
        }
        return result
    }
    
    static func -(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "dimensions of LHS and RHS not matching")
        var result = Self(rows: lhs.rows, columns: lhs.columns)
        for r in 0..<lhs.rows {
            for c in 0..<lhs.columns {
                result[r, c] = lhs[r, c] - rhs[r, c]
            }
        }
        return result
    }
    
    static func *(lhs: Entry, rhs: Self) -> Self {
        var result = Self(rows: rhs.rows, columns: rhs.columns)
        for r in 0..<rhs.rows {
            for c in 0..<rhs.columns {
                result[r, c] = lhs * rhs[r, c]
            }
        }
        return result
    }
    
    static func *(lhs: Self, rhs: Entry) -> Self {
        var result = Self(rows: lhs.rows, columns: lhs.columns)
        for r in 0..<lhs.rows {
            for c in 0..<lhs.columns {
                result[r, c] = lhs[r, c] * rhs
            }
        }
        return result
    }
    
    static func *(lhs: Self, rhs: Self) -> Self {
        assert(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "dimensions of LHS and RHS not matching")
        var result = Self(rows: lhs.rows, columns: lhs.columns)
        for r in 0..<lhs.rows {
            for c in 0..<lhs.columns {
                result[r, c] = lhs[r, c] * rhs[r, c]
            }
        }
        return result
    }
    
    static func •(lhs: Self, rhs: Self) -> Self {
        assert(lhs.columns == rhs.rows, "dimensions of LHS and RHS not matching")
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
