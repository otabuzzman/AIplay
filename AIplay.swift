import SwiftUI
import PlaygroundTester

struct ContentView: View {
    var body: some View {
        VStack {
            AppIcon()
                .frame(width: /*@START_MENU_TOKEN@*/100/*@END_MENU_TOKEN@*/, height: /*@START_MENU_TOKEN@*/100/*@END_MENU_TOKEN@*/, alignment: /*@START_MENU_TOKEN@*/.center/*@END_MENU_TOKEN@*/)
            MNIST()
        } 
    }
}

@main
struct AIplay: App {
    init() {
        PlaygroundTester.PlaygroundTesterConfiguration.isTesting = false
    }
    
    var body: some Scene {
        WindowGroup {
            PlaygroundTester.PlaygroundTesterWrapperView {
                ContentView()
            }
        }
    }
}

struct MNIST: View {
    @StateObject var mnist = MNISTViewModel()
    @State private var folderPickerShow = getAppFolder() == nil
    
    var body: some View {
        Circle().foregroundColor(
            mnist.dataset.count == 0 ? .gray :
                mnist.dataset.count == 4 ? .green :
                    .yellow
        )
        .task {
            mnist.load(from: getAppFolder())
        }
        .sheet (isPresented: $folderPickerShow) {
            FolderPicker { result in
                switch result {
                case .success(let folder):
                    folder.accessSecurityScopedResource { folder in
                        setAppFolder(url: folder)
                    }
                    mnist.load(from: getAppFolder())
                default: // .failure(let error)
                    break
                }
            }
        }
        Text("\(mnist.dataset[.images(.train)] == nil ? 0 : (mnist.dataset[.images(.train)] as! [[Float]]).count)")
        Text("\(mnist.dataset[.images(.test)] == nil ? 0 : (mnist.dataset[.images(.test)] as! [[Float]]).count)")
        Text("\(mnist.dataset[.labels(.train)] == nil ? 0 : (mnist.dataset[.labels(.train)] as! [UInt8]).count)")
        Text("\(mnist.dataset[.labels(.test)] == nil ? 0 : (mnist.dataset[.labels(.test)] as! [UInt8]).count)")
    }
}

internal func setAppFolder(url: URL) {
    if let bookmark = try? url.bookmarkData(options: [/* .withSecurityScope */]) {
        UserDefaults.standard.set(bookmark, forKey: "appFolder")
    }
}

internal func getAppFolder() -> URL? {
    var appFolder: URL?
    if let bookmark = UserDefaults.standard.object(forKey: "appFolder") as? Data {
        var isStale = false
        appFolder = try? URL(
            resolvingBookmarkData: bookmark,
            options: [/* .withSecurityScope */],
            bookmarkDataIsStale: &isStale)
        if isStale { setAppFolder(url: appFolder!) }
    }
    return appFolder
}

struct AppIcon: View {
    var body: some View {
        GeometryReader { dim in
            let w = dim.size.width
            let h = dim.size.height
            ZStack(alignment: .center) {
                Image(systemName: "cpu")
                    .resizable()
                Rectangle()
                    .frame(width: w * 64 / 96, height: h * 64 / 96)
                    .colorInvert()
                Image(systemName: "brain")
                    .resizable()
                    .frame(width: w * 56 / 96, height: h * 56 / 96)
            }
        }
    }
}

struct Network {
    var layers: [Layer]
    let learningRate: Float
    
    // each layer's output O
    private var O: [Matrix<Float>]
    
    mutating func query(for I: Matrix<Float>) -> Matrix<Float> {
        O = [I] // first is output of pseudo input layer, which corresponds to input data
        for layer in layers { // query each layer in turn with output of previous
            O.append(layer.query(for: O.last!))
        }
        return O.last! // network's output layer O
    }
    
    mutating func train(for I: Matrix<Float>, with T: Matrix<Float>) -> Void {
        // network error at output layer O as square of difference T - O
        var E = (T - query(for: I)).map { error in powf(error, 2.0) }
        // back propagate error layer by layer in reverse order
        for layer in (0..<layers.count).reversed() {
            E = layers[layer].train(for: O[layer], with: E, at: learningRate)
        }
    }
}

struct Layer {
    private let inputs: Int
    private let punits: Int
    private let f: (Float) -> Float
    
    private var W: Matrix<Float>
    
    init(
        numberOfInputs inputs: Int = 1,
        numberOfPUnits punits: Int = 1,
        activationFunction f: @escaping (Float) -> Float = { $0 }) {
            
            self.inputs = inputs
            self.punits = punits
            self.f = f
            
            let range = 1.0 / powf(Float(inputs), 0.5)
            W = Matrix<Float>(rows: inputs, columns: punits).map { _ in Float.random(in: -range...range) }
        }
    
    func query(for I: Matrix<Float>) -> Matrix<Float> {
        return (W • I).map { f($0) }
    }
    
    mutating func train(for I: Matrix<Float>, with E: Matrix<Float>, at rate: Float) -> Matrix<Float> {
        let O = query(for: I)
        let B = W.T • E
        W = W + rate * ((E * O * (1.0 - O)) • I.T)
        return B
    }
}
