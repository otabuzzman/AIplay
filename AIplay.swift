import SwiftUI
import PlaygroundTester

struct ContentView: View {
    @State private var showAppInfo = false
    
    var body: some View {
        HStack {
            Label("AIplay", image: "npu")
            Spacer()
            Button {
                showAppInfo.toggle()
            } label: {
                Image(systemName: "info.circle")
            }
        }
        .font(.largeTitle)
        .padding()
        .sheet(isPresented: $showAppInfo) {
            appInfo(isPresented: $showAppInfo)
        }
        MYONNView()
    }
}

func appInfo(isPresented: Binding<Bool>) -> some View {
    let section00 = try! AttributedString(markdown:
            """
            **General**
            AIplay implements a neural network MVP. It predicts numbers from the MNIST dataset as well as handwritten input. The app is a showcase and does not have a configuration interface. Changing parameters is done in source code and therefore requires Xcode or Swift Playgrounds 4 (iPadOS).
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    let section00 = try! AttributedString(markdown:
            """
            **Usage**
            When first started, a file picker dialog opens to select a folder to save and load files.
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    let section01 = try! AttributedString(markdown:
            """
            1. Basic information about the app and usage (this text).
            2. Progress indicator.
            3. Predict a random image from the MNIST test dataset (contains 10000 items).
            4. Prediction result. Turns red to indicate false prediction for test set item.
            5. Display raw image used for prediction from dataset or handwritten.
            6. Predict handwritten number input. Experimental feature. Input position in the sketch area and line width must correspond to MNIST. To get an idea have a look at some images from random MNIST test set prediction (3) in display area (5). To increase line width on larger displays (e.g. iPad or device held landscape) try drawing input multiple times.
            7. Clear sketch area.
            8. Reload the MNIST dataset into memory. Download the files and unzip if necessary. Save files to the app folder selected on first launch.
            9. Per dataset-file state: grey if not present, yellow while loading, red on error and green on success. Circles from left to right correspond to files containing training images, training labels, test images and test labels respectively.
            10. Mini-batch size (hard-coded).
            11. Learning rate (hard-coded).
            12. Number of epochs applied to current network.
            13. Time required to train the most recent epoch.
            14. Train network with next mini-batch from MNIST training set.
            15. Train network with full MNIST training set (contains 60000 items). Expensive in terms of time and battery.
            16. Query network with full MNIST test set and calculate performance.
            17. Network performance (accuracy). Multiply by 100 to get percent.
            18. Save current network (model) in Files app. Proprietary format.
            19. Load network with a model from the Files app. Overwrites current network without warning.
            20. Reset network without warning. Stops training in progress. 
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    for itm in 1...20 {
        section01[section01.range(of: "\(item).")!].foregroundColor = .accentColor
    }
    let section02 = try! AttributedString(markdown:
            """
            **Engine compartment**
            The hard-coded configuration foresees 3 layers with 784 input nodes, 100 hidden nodes and 10 output nodes. The nodes are fully connected. The activation function is sigmoid. Gradient descent is stochastic (SGD) and mini-batch with a size of 30. Training in mini-batch mode leverages any available cores (featuring Swift Concurrency). A GPU implementation for applying the activation function is available (featuring Metal) but deactivated because the network is too small to gain performance benefits from it.
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    let section03 = try! AttributedString(markdown:
            """
            **Soap**
            In the early 1990s I read an article about backpropagation in a computer magazine. The author described the then still new method of Rummelhard et al. (1986) and implemented it with Turbo Pascal. I typed the program into my 386 and today forgot what came out. But I stayed true to the overarching theme, and when it became popular again in the 2010s with the availability of cheap, powerful GPUs, the idea of doing something with AI arose. Practical. I had read some papers and books about neural networks and machine learning and was fascinated by the simplicity of the underlying concepts - as far as I understood. With Swift Playgrounds 4 for iPadOS, a kind of lightweight Xcode for app development on the iPad, the barrier became very low that I finally started...
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    return infoView(isPresented: isPresented, VStack {
        Text(section00)
        Text(section01)
        Text(section02)
        Text(section03)
    })
}

func infoView(isPresented: Binding<Bool>, _ description: Text) -> some View {
    VStack {
        HStack {
            Text("DESCRIPTION")
                .font(.headline)
                .padding()
            Spacer()
        }
        description
            .lineLimit(nil)
        Button {
            isPresented.wrappedValue.toggle()
        } label: {
            Text("Close")
        }
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

@main
struct AIplay: App {
    @State private var folderPickerShow = getAppFolder() == nil

    init() {
        PlaygroundTester.PlaygroundTesterConfiguration.isTesting = false
    }
    
    var body: some Scene {
        WindowGroup {
            PlaygroundTester.PlaygroundTesterWrapperView {
                ContentView()
            }
            .sheet(isPresented: $folderPickerShow) {
                FolderPicker { result in
                    switch result {
                    case .success(let folder):
                        folder.accessSecurityScopedResource { folder in
                            setAppFolder(url: folder)
                        }
                    default: // .failure(let error)
                        break
                    }
                }
            }
        }
    }
}

func stringOfElements<T>(in this: [T], count: Int? = nil, format: @escaping (T) -> String = { element in String(describing: element) }) -> String {
    var stringOfElements = ""
    let substring: (Int, Int) -> String = { startIndex, count in
        var substring: String = ""
        let endIndex = startIndex + count - 1
        for index in startIndex...endIndex {
            substring += format(this[index])
            if index == endIndex { break }
            substring += ", "
        }
        return substring
    }
    let maxCount = count ?? this.count
    if this.count > maxCount {
        stringOfElements += substring(0, maxCount - 2) + ", ... " + substring(maxCount  - 2, 2)
    } else {
        stringOfElements += substring(0, maxCount)
    }
    return stringOfElements
}

// https://holyswift.app/create-generic-factory-in-swift/
struct GenericFactory {
    static func create<Config, Output, Factory: AbstractFactory>(_ object: Factory, _ config: Config) -> Output? where Factory.Config == Config, Factory.Output == Output {
        object.create(config)
    }
}

protocol AbstractFactory {
    associatedtype Config
    associatedtype Output
    func create(_ config: Config) -> Output?
}
