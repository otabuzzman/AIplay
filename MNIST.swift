import Foundation
import DataCompression

enum MNISTItem: Hashable {
    enum Subset: String {
        case train = "train"
        case test = "t10k"
    }
    
    case images(_ subset: Subset)
    case labels(_ subset: Subset)
    
    var name: String {
        switch self {
        case .images(let subset):
            return subset.rawValue + "-images-idx3-ubyte"
        case .labels(let subset):
            return subset.rawValue + "-labels-idx1-ubyte"
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

class MNIST: ObservableObject {
    private var state = NSLock()
    
    @Published private(set) var dataset: [MNISTItem : Any] = [:]
    
    func load(from folder: URL?, _ completion: ((Error?) -> Void)? = nil) {
        guard
            let folder = folder
        else { return }
        let load = { [self] (item: MNISTItem, itemUrl: URL) -> Void in
            Task {
                switch item {
                case .images:
                    let images = Self.readImages(from: itemUrl)
                    synchronize { dataset[item] = images }
                case .labels:
                    let labels = Self.readLabels(from: itemUrl)
                    synchronize { dataset[item] = labels }
                }
            }
        }
        let baseURL = "http://yann.lecun.com/exdb/mnist/"
        let items: [MNISTItem] = [
            .images(.train), .labels(.train),
            .images(.test), .labels(.test)
        ]
        for item in items {
            let itemUrl = folder.appending(path: item.name)
            if FileManager.default.fileExists(atPath: itemUrl.path) {
                load(item, itemUrl)
                continue
            }
            let itemGzUrl = itemUrl.appendingPathExtension(for: .gzip)
            if !FileManager.default.fileExists(atPath: itemGzUrl.path) {
                let remoteItemUrl = URL(string: baseURL)?
                    .appending(path: item.name)
                    .appendingPathExtension(for: .gzip)
                Self.download(source: remoteItemUrl!, target: itemGzUrl) { error in
                    if let error = error {
                        completion?(error)
                        return
                    }
                    do {
                        try Self.gunzip(source: itemGzUrl, target: itemUrl)
                        load(item, itemUrl)
                    } catch {
                        completion?(error)
                        return
                    }
                }
            } else {
                do {
                    try Self.gunzip(source: itemGzUrl, target: itemUrl)
                    load(item, itemUrl)
                } catch {
                    completion?(error)
                    return
                }
            }
        }
    }
    
    private static func gunzip(source: URL, target: URL) throws {
        let contentGz = try Data(contentsOf: source)
        guard
            let content = contentGz.gunzip()
        else { throw MNISTError.gunzip }
        try content.write(to: target, options: .noFileProtection)
    }
    
    private static func download(source: URL, target: URL, _ completion: @escaping (Error?) -> Void) {
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
    
    private static func readImages(from source: URL) -> [[UInt8]] {
        let handle = try! FileHandle(forReadingFrom: source)
        var images = [[UInt8]]()
        
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
    
    private static func readLabels(from source: URL) -> [UInt8] {
        let handle = try! FileHandle(forReadingFrom: source)
        
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
        state.lock()
        defer { state.unlock() }
        return try code()
    }
}

extension URL {
    func accessSecurityScopedResource(_ accessor: (URL) -> Void) {
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
