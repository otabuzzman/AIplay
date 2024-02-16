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

struct NNXD {
    // header
    let magic = "!NNXD"
    let version = 2
    
    // section: hyper params
    var epochsWanted: Int
    var miniBatchSize: Int
    
    // section: network
    var network: Network
    
    // section: measures
    var measures: Array<Measures>
}

extension NNXD {
    var config: NetworkConfig {
        get {
            var config = network.config
            config.epochsWanted = epochsWanted
            config.miniBatchSize = miniBatchSize
            return config
        }
        set(config) {
            network = GenericFactory.create(NetworkFactory(), config)!
            epochsWanted = config.epochsWanted
            miniBatchSize = config.miniBatchSize
        }
    }
}
extension NNXD: CustomCoder {
    var encode: Data {
        // header
        var data = magic.data(using: .utf8)! // cannot use .encode here
        data += version.encode
        
        // section: hyper params
        data += epochsWanted.encode
        data += miniBatchSize.encode
        
        // section: network
        data += network.encode
        
        // section: measures
        data += measures.count.encode
        measures.forEach { data += $0.encode }
        
        return data
    }
    
    init?(from: Data) {
        var data = from
        
        // check magic string... (cannot use .init? here)
        guard let magic = String(data: data[..<magic.count], encoding: .utf8) else { return nil }
        if magic != self.magic { return nil }
        data = data.advanced(by: magic.count)
        // ...and version
        guard let version = Int(from: data) else { return nil }
        if version != self.version { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        // section: hyper params
        guard let epochsWanted = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        guard let miniBatchSize = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        // section: network
        guard let networkSize = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        guard let network = Network(from: data) else { return nil }
        data = data.advanced(by: networkSize)
        
        // section: measures
        guard let measuresCount = Int(from: data) else { return nil }
        data = data.advanced(by: MemoryLayout<Int>.size)
        
        measures = [Measures]()
        for _ in 0..<measuresCount {
            guard let measureSize = Int(from: data) else { return nil }
            data = data.advanced(by: MemoryLayout<Int>.size)
            guard let element = Measures(from: data) else { return nil }
            data = data.advanced(by: measureSize)
            measures.append(element)
        }
        
        self.epochsWanted = epochsWanted
        self.miniBatchSize = miniBatchSize
        self.network = network
    }
}
