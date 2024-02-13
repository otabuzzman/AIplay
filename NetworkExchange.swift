import SwiftUI

import UniformTypeIdentifiers

struct NetworkExchangeDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.nnxd] }
    static var writableContentTypes: [UTType] { [.nnxd] }
    
    private var content: Data
    
    init(_ content: Data) {
        self.content = content
    }
    
    init(configuration: FileDocumentReadConfiguration) throws {
        guard
            let content = configuration.file.regularFileContents
        else {
            let url = URL(string: configuration.file.filename ?? "unknown")!
            throw NetworkExchangeDocumentError.modified(url)
        }
        self.content = content
    }
    
    func fileWrapper(configuration: FileDocumentWriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: content)
    }
}

enum NetworkExchangeDocumentError: Error {
    case modified(URL)
}

extension NetworkExchangeDocumentError {
    var description: String {
        switch self {
        case .modified(let url):
            return "NNXD file \(url) modified while reading"
        }
    }
}

enum NetworkExchangeError: Error {
    case nnxdSave(Error)
    case nnxdLoad(Error)
    case nnxdRead(URL, Error)
    case nnxdDecode(URL)
}

extension NetworkExchangeError {
    var description: String {
        switch self {
        case .nnxdSave(let error):
            return "save NNXD file failed: \(error)"
        case .nnxdLoad(let error):
            return "load NNXD file failed: \(error)"
        case .nnxdRead(let url, let error):
            return "read NNXD file \(url) failed: \(error)"
        case .nnxdDecode(let url):
            return "decode NNXD file \(url) failed"
        }
    }
}

extension UTType {
    public static let nnxd = UTType(exportedAs: "com.otabuzzman.aiplay.nnxd")
}
