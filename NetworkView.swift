import SwiftUI
import UniformTypeIdentifiers

enum NetworkViewError: Error {
    case nndxSave(Error)
    case nndxLoad(Error)
    case nndxRead(URL, Error)
    case nndxDecode(URL)
}

extension NetworkViewError {
    var description: String {
        switch self {
        case .nndxSave(let error):
            return "save NNDX file failed with error \(error)"
        case .nndxLoad(let error):
            return "load NNDX file failed with error \(error)"
        case .nndxRead(let url, let error):
            return "read NNDX file \(url) failed with error \(error)"
        case .nndxDecode(let url):
            return "decode NNDX file \(url) failed"
        }
    }
}

struct NetworkView: View {
    @ObservedObject var viewModel: NetworkViewModel
    
    @State private var error: NetworkViewError? = nil
    
    @State private var isExporting = false
    @State private var isImporting = false
    
    var body: some View {
        VStack {
            Circle().foregroundColor(.gray)
            Button {
                isExporting = true
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .fileExporter(isPresented: $isExporting,
                          document: NetworkExchangeDocument(viewModel.network.encode),
                          contentType: .nnxd, defaultFilename: "Untitled") { result in
                switch result {
                case .success:
                    break
                case .failure(let error):
                    self.error = .nndxSave(error)
                }
            }
            Button {
                isImporting = true
            } label: {
                Image(systemName: "square.and.arrow.up")
            }
            .fileImporter(isPresented: $isImporting,
                          allowedContentTypes: [.nnxd], allowsMultipleSelection: false) { result in
                switch result {
                case .success(let url):
                    do {
                        let content = try Data(contentsOf: url[0])
                        guard
                            let network = Network(from: content)
                        else {
                            self.error = .nndxDecode(url[0])
                            return
                        }
                        viewModel.set(network: network)
                    } catch {
                        self.error = .nndxRead(url[0], error)
                    }
                case .failure(let error):
                    self.error = .nndxLoad(error)
                }
            }
        }
    }
}

enum NetworkExchangeDocumentError: Error {
    case modified
}

extension NetworkExchangeDocumentError {
    var description: String {
        switch self {
        case .modified:
            return "document modified while reading"
        }
    }
}

extension UTType {
    public static let nnxd = UTType(exportedAs: "com.otabuzzman.aiplay.nnxd")
}

struct NetworkExchangeDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.nnxd] }
    static var writableContentTypes: [UTType] { [.nnxd] }
    
    private(set) var content: Data
    
    init(_ content: Data) {
        self.content = content
    }
    
    init(configuration: FileDocumentReadConfiguration) throws {
        guard
            let content = configuration.file.regularFileContents
        else { throw NetworkExchangeDocumentError.modified }
        self.content = content
    }
    
    func fileWrapper(configuration: FileDocumentWriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: content)
    }
}

class NetworkViewModel: ObservableObject {
    private(set) var network: Network!
    
    init(_ network: Network) {
        set(network: network)
    }
    
    func set(network: Network) {
        self.network = network
    }
    
    func query(for I: [UInt8]) -> Matrix<Float> {
        let input = Matrix<Float>(rows: I.count, columns: 1, entries: I.map({ Float($0) }))
            .map { ($0 / 255.0 * 0.99) + 0.01 } // MYONN, p. 151 ff.
        return network.query(for: input)
    }
    
    func train(for I: [UInt8], with T: UInt8) -> Void {
        let input = Matrix<Float>(rows: I.count, columns: 1, entries: I.map({ Float($0) }))
            .map { ($0 / 255.0 * 0.99) + 0.01 }
        var target = Matrix<Float>(rows: 10, columns: 1)
            .map { _ in 0.01 }
        target[Int(T), 0] = 0.99
        network.train(for: input, with: target)
    }
}
