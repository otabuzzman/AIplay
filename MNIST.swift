import SwiftUI
import Foundation

import DataCompression

enum MNISTDatasetItem: Hashable {
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

enum MNISTDatasetError: Error {
    case gunzip
    case response(code: Int)
}

struct MNISTDataset: View {
    private var state = NSLock()
    
    @State private var folderPickerShow = getAppFolder() == nil
    @State private(set) var files: [MNISTDatasetItem : Any] = [:]
    
    var body: some View {
        Circle().foregroundColor(
            files.count == 0 ? .gray :
                files.count == 4 ? .green :
                    .yellow
        )
        .task {
            load(from: getAppFolder())
        }
        .sheet (isPresented: $folderPickerShow) {
            FolderPicker { result in
                switch result {
                case .success(let folder):
                    folder.accessSecurityScopedResource { folder in
                        setAppFolder(url: folder)
                    }
                    load(from: getAppFolder())
                default: // .failure(let error)
                    break
                }
            }
        }
        Text("\(files[.images(.train)] == nil ? 0 : (files[.images(.train)] as! [[Float]]).count)")
        Text("\(files[.images(.test)] == nil ? 0 : (files[.images(.test)] as! [[Float]]).count)")
        Text("\(files[.labels(.train)] == nil ? 0 : (files[.labels(.train)] as! [UInt8]).count)")
        Text("\(files[.labels(.test)] == nil ? 0 : (files[.labels(.test)] as! [UInt8]).count)")
    }
}

extension MNISTDataset {
    private static let baseURL = "http://yann.lecas!.com/exdb/mnist/"
    
    private func load(from folder: URL?, _ completion: ((Error?) -> Void)? = nil) {
        guard
            let folder = folder
        else { return }
        let items: [MNISTDatasetItem] = [.images(.train), .labels(.train), .images(.test), .labels(.test)]
        for item in items {
            let itemUrl = folder.appending(path: item.name)
            if !FileManager.default.fileExists(atPath: itemUrl.path) {
                let itemGzUrl = itemUrl.appendingPathExtension(for: .gzip)
                if !FileManager.default.fileExists(atPath: itemGzUrl.path) {
                    let remoteItemUrl = URL(string: Self.baseURL)?
                        .appending(path: item.name)
                        .appendingPathExtension(for: .gzip)
                    Self.download(source: remoteItemUrl!, target: itemGzUrl) { error in
                        if let error = error {
                            completion?(error)
                            return
                        }
                        do {
                            try Self.gunzip(source: itemGzUrl, target: itemUrl)
                        } catch {
                            completion?(error)
                            return
                        }
                    }
                } else {
                    do {
                        try Self.gunzip(source: itemGzUrl, target: itemUrl)
                    } catch {
                        completion?(error)
                        return
                    }
                }
            }
            switch item {
            case .images:
                Task {
                    let images = Self.readImages(from: itemUrl)
                    synchronize { files[item] = images }
                }
            case .labels:
                Task {
                    let labels = Self.readLabels(from: itemUrl)
                    synchronize { files[item] = labels }
                }
            }
        }
    }
    
    private static func gunzip(source: URL, target: URL) throws {
        let contentGz = try Data(contentsOf: source)
        guard
            let content = contentGz.gunzip()
        else { throw MNISTDatasetError.gunzip }
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
                    completion(MNISTDatasetError.response(code: statusCode))
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
    
    private static func readImages(from source: URL) -> [[Float]] {
        let handle = try! FileHandle(forReadingFrom: source)
        var images = [[Float]]()
        
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
            var imageData = Array<Float>(repeating: 0, count: imageSize)
            _ = imageData.withUnsafeMutableBytes { rawImageData.copyBytes(to: $0) }
            images.append(imageData.map { Float($0) / Float(255) })
        }
        
        return images
    }
    
    private static func readLabels(from source: URL) -> [UInt8] {
        let handle = try! FileHandle(forReadingFrom: source)
        
        handle.seek(toFileOffset: 0)
        _ = handle.readData(ofLength: MemoryLayout<UInt32>.size) // magic number
        
        let rawNumberOfItems = handle.readData(ofLength: MemoryLayout<UInt32>.size)
        let numberOfItems = UInt32(bigEndian: rawNumberOfItems.withUnsafeBytes({ $0.load(as: UInt32.self) }))

        let rawLabelsData = handle.readDataToEndOfFile()
        var labels = Array<UInt8>(repeating: 0, count: Int(numberOfItems))
        _ = labels.withUnsafeMutableBytes { rawLabelsData.copyBytes(to: $0) }
        
        return labels
    }
    
    // https://stackoverflow.com/a/66228135 (comprehensive refresher on capture lists and thread safety)
    func synchronize<T>(code: () throws -> T) rethrows -> T {
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
