import SwiftUI
import Foundation

import DataCompression

struct MNISTView: View {
    @ObservedObject private var viewModel = MNISTViewModel(in: getAppFolder())
    
    var body: some View {
        HStack {
            VStack {
                Circle().foregroundColor(viewModel.state[.images(.train)]?.color)
                Circle().foregroundColor(viewModel.state[.labels(.train)]?.color)
            }
            VStack {
                Circle().foregroundColor(viewModel.state[.images(.test)]?.color)
                Circle().foregroundColor(viewModel.state[.labels(.test)]?.color)
            }
			NetworkView(dataset: viewModel)
        }
    }
}

protocol MNISTEntity { }
typealias MNISTImage = [UInt8]
typealias MNISTLabel = UInt8
extension MNISTImage: MNISTEntity { }
extension MNISTLabel: MNISTEntity { }

enum MNISTSubset: Hashable {
    enum Purpose: String {
        case train = "train"
        case test = "t10k"
    }
    
    case images(Purpose)
    case labels(Purpose)
    
    var name: String {
        switch self {
        case .images(let purpose):
            return purpose.rawValue + "-images-idx3-ubyte"
        case .labels(let purpose):
            return purpose.rawValue + "-labels-idx1-ubyte"
        }
    }
    
    static var all: [MNISTSubset] { [.images(.train), .labels(.train), .images(.test), .labels(.test)] }
}

enum MNISTState {
    case missing
    case loading
    case loaded
    case failed(Error)
    
    var color: Color {
        switch self {
        case .missing:
            return .gray
        case .loading:
            return .yellow
        case .loaded:
            return .green
        case .failed:
            return .red
        }
    }
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

extension MNISTView {
    class MNISTViewModel: ObservableObject {
        private var lock = NSLock()
        
        @Published private(set) var subsets: [MNISTSubset : [MNISTEntity]] = [:]
        @Published private(set) var state: [MNISTSubset : MNISTState] = [:]
        
        init(in folder: URL?) {
            MNISTSubset.all.forEach { state[$0] = .missing }
            if let folder = folder {
                load(from: folder)
            }
        }
        
        func load(from folder: URL) -> Void {
            let baseURL = "http://yann.lecun.com/exdb/mnist/"
            let load = { [self] (subset: (MNISTSubset, URL), error: Error?) -> Void in
                if let error = error {
                    synchronize { state[subset.0] = .failed(error) }
                }
                Task { @MainActor in
                    do {
                        let entity: [MNISTEntity]
                        switch subset.0 {
                        case .images:
                            entity = try Self.readImages(from: subset.1)
                        case .labels:
                            entity = try Self.readLabels(from: subset.1)
                        }
                        synchronize {
                            subsets[subset.0] = entity
                            state[subset.0] = .loaded
                        }
                    } catch {
                        state[subset.0] = .failed(error)
                    }
                }
            }
            MNISTSubset.all.forEach { subset in
                synchronize { state[subset] = .loading }
                let itemUrl = folder.appending(path: subset.name)
                if FileManager.default.fileExists(atPath: itemUrl.path) {
                    load((subset, itemUrl), nil)
                    return
                }
                let itemGzUrl = itemUrl.appendingPathExtension(for: .gzip)
                if !FileManager.default.fileExists(atPath: itemGzUrl.path) {
                    let remoteItemUrl = URL(string: baseURL)?
                        .appending(path: subset.name)
                        .appendingPathExtension(for: .gzip)
                    Self.download(source: remoteItemUrl!, target: itemGzUrl) { error in
                        if let error = error {
                            load((subset, itemUrl), error)
                            return
                        }
                        do {
                            try Self.gunzip(source: itemGzUrl, target: itemUrl)
                            load((subset, itemUrl), nil)
                        } catch {
                            load((subset, itemUrl), error)
                            return
                        }
                    }
                } else {
                    do {
                        try Self.gunzip(source: itemGzUrl, target: itemUrl)
                        load((subset, itemUrl), nil)
                    } catch {
                        load((subset, itemUrl), error)
                        return
                    }
                }
            }
        }
        
        private static func gunzip(source: URL, target: URL) throws -> Void {
            let contentGz = try Data(contentsOf: source)
            guard
                let content = contentGz.gunzip()
            else { throw MNISTError.gunzip }
            try content.write(to: target, options: .noFileProtection)
        }
        
        private static func download(source: URL, target: URL, _ completion: @escaping (Error?) -> Void) -> Void {
            let urlSession = URLSession(configuration: .default)
            let task = urlSession.downloadTask(with: source) { location, response, error in
                if let error = error {
                    completion(error)
                    return
                }
                if let response = response {
                    let statusCode = (response as! HTTPURLResponse).statusCode
                    if statusCode != 200 {
                        completion(MNISTError.response(code: statusCode))
                        return
                    }
                }
                guard
                    let location = location
                else {
                    completion(URLError(.badURL))
                    return
                }
                do {
                    try FileManager.default.copyItem(at: location, to: target)
                } catch {
                    completion(error)
                    return
                }
                completion(nil)
            }
            task.resume()
        }
        
        private static func readImages(from source: URL) throws -> [MNISTEntity] {
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
        
        private static func readLabels(from source: URL) throws -> [MNISTEntity] {
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
