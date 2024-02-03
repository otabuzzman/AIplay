import SwiftUI
import Foundation

import DataCompression

struct MNISTView: View {
    @ObservedObject var viewModel: MNISTViewModel
    @Binding var ready: Bool
    @Binding var error: Bool

    @State private var showFolderPicker = getAppFolder() == nil
    
    @State private var showLoadError = false
    
    typealias LoadError = (subset: MNISTSubset, error: Error)
    @State private var loadError: LoadError?
    
    private func load() -> Void {
        Task {
            guard
                let folder = getAppFolder()
            else { return }
            
            ready = false
            error = false
            
            await viewModel.load(from: folder)
            MNISTSubset.allCases.forEach {
                switch viewModel.state[$0] {
                case .failed:
                    error = true
                default: // .loaded
                    ready = true
                }
            }
        }
    }
    
    var body: some View {
        HStack {
            HStack { // mimic Button with limited tap area
                Image(systemName: "arrow.counterclockwise.icloud")
                Text("Reload MNIST")
            }
            .foregroundColor(.accentColor)
            .contentShape(Rectangle())
                .onTapGesture {
                    load()
                }
            Spacer()
            HStack(spacing: 4) {
                ForEach(MNISTSubset.allCases, id: \.self) { subset in
                    switch viewModel.state[subset] {
                    case .missing, .none:
                        Image(systemName: "multiply.circle").foregroundColor(.gray)
                    case .loading:
                        Image(systemName: "clock").foregroundColor(.yellow)
                    case .loaded:
                        Image(systemName: "info.circle").foregroundColor(.green)
                    case .failed(let error):
                        Image(systemName: "checkmark.circle")
                            .foregroundColor(.red)
                            .contentShape(Circle())
                            .onTapGesture {
                                loadError = (subset, error)
                                showLoadError = true
                            }
                    }
                }
                .font(.title3)
            }
        }
        .onFirstAppear {
            load()
        }
        .sheet(isPresented: $showFolderPicker) {
            FolderPicker { result in
                switch result {
                case .success(let folder):
                    folder.accessSecurityScopedResource { folder in
                        setAppFolder(url: folder)
                    }
                    load()
                default: // .failure(let error)
                    break
                }
            }
        }
        .alert("Error loading :\n\(loadError?.subset.file ?? "nil")", isPresented: $showLoadError) {} message: {
            if let error = loadError?.error {
                let type = "Caught \(String(describing: type(of: error))) exception"
                switch error {
                case let error as MNISTError:
                    Text("\(type) :\n\(error.description)")
                default: // Error
                    Text("\(type) :\n\(error.localizedDescription)")
                }
            } else {
                Text("Caught nil exception :\ncan’t actually happen")
            }
        }
    }
}

enum MNISTSubset: Hashable {
    enum Purpose: String {
        case train = "train"
        case test = "t10k"
    }
    
    case images(Purpose)
    case labels(Purpose)
    
    var file: String {
        switch self {
        case .images(let purpose):
            return purpose.rawValue + "-images-idx3-ubyte"
        case .labels(let purpose):
            return purpose.rawValue + "-labels-idx1-ubyte"
        }
    }
    
    static var allCases: [Self] { [.images(.train), .labels(.train), .images(.test), .labels(.test)] }
}

protocol MNISTItem { }
typealias MNISTImage = [UInt8]
typealias MNISTLabel = UInt8
extension MNISTImage: MNISTItem { }
extension MNISTLabel: MNISTItem { }

enum MNISTState {
    case missing
    case loading
    case loaded
    case failed(any Error)
}

enum MNISTError: Error {
    case gunzip
    case response(code: Int)
}

extension MNISTError {
    var description: String {
        switch self {
        case .gunzip:
            return "gunzip Data failed"
        case .response(let code):
            return "HTTP response returned \(code)"
        }
    }
}

class MNISTViewModel: ObservableObject {
    private var lock = NSLock()
    
    private var trainsetIndex = Array(0..<60000)
    private var items: [MNISTSubset : [MNISTItem]] = [:]
    @Published private(set) var state: [MNISTSubset : MNISTState] = [:]
    
    init() {
        MNISTSubset.allCases.forEach { state[$0] = .missing }
    }
    
    func count(in: MNISTSubset.Purpose) -> Int {
        `in` == .train ? 60000 : 10000
    }
    
    func fetch(_ index: Int, from: MNISTSubset.Purpose) -> (MNISTImage, MNISTLabel) {
        let i = from == .train ? trainsetIndex[index] : index
        let image = (items[.images(from)] as! [MNISTImage])[i]
        let label = (items[.labels(from)] as! [MNISTLabel])[i]
        return (image, label)
    }
    
    func fetch(_ range: Range<Int>, from: MNISTSubset.Purpose) -> ([MNISTImage], [MNISTLabel]) {
        var images: [MNISTImage]
        var labels: [MNISTLabel]
        if from == .train {
            images = range.map { (items[.images(from)] as! [MNISTImage])[trainsetIndex[$0]] }
            labels = range.map { (items[.labels(from)] as! [MNISTLabel])[trainsetIndex[$0]] }
        } else {
            images = (items[.images(from)] as! [MNISTImage])[range].map { $0 }
            labels = (items[.labels(from)] as! [MNISTLabel])[range].map { $0 }
        }
        return (images, labels)
    }
    
    func shuffle() -> Void {
        trainsetIndex.shuffle()
    }
    
    func load(from folder: URL) async -> Void {
        let baseURL = "http://yann.lecun.com/exdb/mnist/"
        MNISTSubset.allCases.forEach { state[$0] = .missing }
        
        await withTaskGroup(of: (MNISTSubset, MNISTState).self, returning: Void.self) { group in 
            MNISTSubset.allCases.forEach { subset in
                group.addTask { [self] in
                    synchronize { state[subset] = .loading }
                    
                    do {
                        let itemUrl = folder.appending(path: subset.file)
                        if FileManager.default.fileExists(atPath: itemUrl.path) {
                            try await load(subset, from: itemUrl)
                            return (subset, .loaded)
                        }
                        
                        let itemGzUrl = itemUrl.appendingPathExtension(for: .gzip)
                        if FileManager.default.fileExists(atPath: itemGzUrl.path) {
                            try Self.gunzip(source: itemGzUrl, target: itemUrl)
                            try await load(subset, from: itemUrl)
                            return (subset, .loaded)
                        }
                        
                        let remoteItemUrl = URL(string: baseURL)?
                            .appending(path: subset.file)
                            .appendingPathExtension(for: .gzip)
                        try await Self.download(source: remoteItemUrl!, target: itemGzUrl) { _ in 
                            try Self.gunzip(source: itemGzUrl, target: itemUrl)
                            try await load(subset, from: itemUrl)
                        }
                        return (subset, .loaded)
                    } catch {
                        synchronize { items[subset] = nil }
                        return (subset, .failed(error))
                    }
                }
            }
            
            for await result in group {
                synchronize { state[result.0] = result.1 }
            }
        }
    }
    
    private static func download(source: URL, target: URL, completion: (URLResponse) async throws -> Void) async throws -> Void {
        let session = URLSession.shared
        let (tmpUrl, response) = try await session.download(from: source)
        let statusCode = (response as! HTTPURLResponse).statusCode
        if statusCode != 200 {
            throw MNISTError.response(code: statusCode)
        }
        try FileManager.default.copyItem(at: tmpUrl, to: target)
        try FileManager.default.removeItem(at: tmpUrl)
        try await completion(response)
    }
    
    private static func gunzip(source: URL, target: URL) throws -> Void {
        let contentGz = try Data(contentsOf: source)
        guard
            let content = contentGz.gunzip()
        else { throw MNISTError.gunzip }
        try content.write(to: target, options: .noFileProtection)
    }
    
    private func load(_ subset: MNISTSubset, from itemUrl: URL) async throws -> Void {
        let item: [MNISTItem]
        switch subset {
        case .images:
            item = try Self.readImages(from: itemUrl)
        case .labels:
            item = try Self.readLabels(from: itemUrl)
        }
        synchronize { items[subset] = item }
    }
    
    private static func readImages(from source: URL) throws -> [MNISTItem] {
        let handle = try FileHandle(forReadingFrom: source)
        var images = [MNISTImage]()
        
        handle.seek(toFileOffset: 0)
        _ = handle.readData(ofLength: MemoryLayout<UInt32>.size) // magic number
        
        let rawNumberOfImages = handle.readData(ofLength: MemoryLayout<UInt32>.size)
        let numberOfImages = UInt32(bigEndian: rawNumberOfImages.withUnsafeBytes({ $0.load(as: UInt32.self) }))
        let rawNumberOfRows = handle.readData(ofLength: MemoryLayout<UInt32>.size)
        let numberOfRows = UInt32(bigEndian: rawNumberOfRows.withUnsafeBytes({ $0.load(as: UInt32.self) }))
        let rawNumberOfColumns = handle.readData(ofLength: MemoryLayout<UInt32>.size)
        let numberOfColumns = UInt32(bigEndian: rawNumberOfColumns.withUnsafeBytes({ $0.load(as: UInt32.self) }))
        
        let imageSize = MemoryLayout<UInt8>.size * Int(numberOfRows * numberOfColumns)
        for _ in 0..<numberOfImages {
            let rawImageData = handle.readData(ofLength: imageSize)
            let imageData = rawImageData.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }
            images.append(imageData)
        }
        
        return images
    }
    
    private static func readLabels(from source: URL) throws -> [MNISTItem] {
        let handle = try FileHandle(forReadingFrom: source)
        
        handle.seek(toFileOffset: 0)
        _ = handle.readData(ofLength: MemoryLayout<UInt32>.size) // magic number
        
        let rawNumberOfItems = handle.readData(ofLength: MemoryLayout<UInt32>.size)
        _ = UInt32(bigEndian: rawNumberOfItems.withUnsafeBytes({ $0.load(as: UInt32.self) }))
        
        let rawLabelsData = handle.readDataToEndOfFile()
        let labels = rawLabelsData.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }
        
        return labels
    }
    
    // https://stackoverflow.com/a/66228135 (comprehensive refresher on capture lists and thread safety)
    private func synchronize<T>(code: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try code()
    }
}

extension URL {
    func accessSecurityScopedResource(_ accessor: (URL) -> Void) -> Void {
        let didStartAccessing = startAccessingSecurityScopedResource()
        defer { if didStartAccessing { stopAccessingSecurityScopedResource() } }
        accessor(self)
    }
    
    // https://developer.apple.com/documentation/foundation/nsurl#1663783
    func obtainSecurityScopedResource() -> URL? {
        var securityScopedUrl: URL?
        if let bookmark = try? self.bookmarkData(options: [/* .withSecurityScope */]) {
            var isStale = false
            securityScopedUrl = try? URL(
                resolvingBookmarkData: bookmark,
                options: [/* .withSecurityScope */],
                bookmarkDataIsStale: &isStale)
        }
        return securityScopedUrl
    }
}

struct FirstAppearModifier: ViewModifier {
    private let perform: () -> Void
    
    @State private var hasAppeared = false
    
    public init(_ perform: @escaping () -> Void) {
        self.perform = perform
    }
    
    public func body(content: Content) -> some View {
        content
            .onAppear {
                guard
                    !hasAppeared
                else { return }
                hasAppeared = true
                perform()
            }
    }
}

extension View {
    func onFirstAppear(_ perform: @escaping () -> Void) -> some View {
        modifier(FirstAppearModifier(perform))
    }
}
