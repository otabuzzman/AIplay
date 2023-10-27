import SwiftUI
import PlaygroundTester

struct ContentView: View {
    @State private var appInfoSow = false
    
    var body: some View {
        HStack {
            Label("AIplay", image: "npu")
            Spacer()
            Button {
                appInfoSow.toggle()
            } label: {
                Image(systemName: "info.circle")
            }
        }
        .font(.largeTitle)
        .padding()
        .sheet(isPresented: $appInfoSow) {
            appInfo(isPresented: $appInfoSow)
                .frame(minWidth: 0, maxWidth: 512)
        }
        MYONNView()
    }
}

func appInfo(isPresented: Binding<Bool>) -> some View {
    let section00 = try! AttributedString(markdown:
            """
            AIplay implements a neural network MVP. It queries the network (prediction) with number images from the MNIST dataset. Handwritten number entries using finger or Apple Pencil are also supported. The app is a showcase. There is no configuration interface. Changing parameters must therefore be done in source code and requires Xcode or Swift Playgrounds 4 (iPadOS).
            """, options: .init(interpretedSyntax: .full))
    var section01 = try! AttributedString(markdown:
            """
            **Usage**
            When you first start it, a file selection dialog opens where you once must select a folder to save and load files. The app then loads the MNIST data set from the internet. If the four icons (11) are green, you can start. Click red ones in case of errors to get details.
            
            Place the `.nndx` files from the [repository](https://github.com/otabuzzman/AIplay/Resources) in the selected folder on your device and load (20) one of the pre-defined models or start right away training a new one.
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    
    section01[section01.range(of: "11")!].foregroundColor = .accentColor
    section01[section01.range(of: "20")!].foregroundColor = .accentColor
    
    let section02 = try! AttributedString(markdown:
            """
            The image shows an idealized app view. Actual proportions and visibilty of controls vary depending on device and orientation.
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    
    var section03 = try! AttributedString(markdown:
            """
            1. Basic information and usage (this text).
            2. Progress indicator.
            3. Query with random image from MNIST test set (contains 10000 items).
            4. Prediction result. Turns red to indicate false prediction for test set item.
            5. Open/ close network result details view.
            6. Detailed propabillities of network result.
            7. Display raw image of query input from dataset or handwritten.
            8. Query with handwritten number input. Experimental feature. Input position in the sketch area and line width must correspond to MNIST. To get an idea have a look at some images (7) when performing random MNIST test set queries (3). To increase line width on larger displays (e.g. iPad or device held landscape) try drawing input multiple times.
            9. Clear sketch area.
            10. Reload the MNIST dataset into memory. Downloads the files and unzips them if necessary. Saves files in the app folder selected on first launch.
            11. Per dataset-file state. Grey: present, yellow: loading, red: error and green: success. Circles from left to right correspond to files containing training images, training labels, test images and test labels respectively.
            12. Mini-batch size (hard-coded).
            13. Learning rate (hard-coded).
            14. Number of trainings with full training set (contains 60000 items) applied to current network.
            15. Time required to train the most recent epoch.
            16. Train network with next mini-batch from MNIST training set.
            17. Train network with full MNIST training set. Takes a couple of minutes.
            18. Query network with full MNIST test set and calculate accuracy.
            19. Network accuracy. Multiply by 100 to get percent.
            20. Load network with a model from the Files app. Overwrites current network without warning.
            21. Save current network (model) in Files app in a roprietary format.
            22. Reset network without warning. Stops training in progress.
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    
    for item in 1...22 {
        section03[section03.range(of: "\(item).")!].foregroundColor = .accentColor
    }
    
    let section04 = try! AttributedString(markdown:
            """
            **Under the hood**
            The hard-coded configuration foresees 3 layers with 784 input nodes, 100 hidden nodes and 10 output nodes. The nodes are fully connected. The activation function is sigmoid. Gradient descent is stochastic (SGD) and mini-batch with a size of 30. Training in mini-batch mode leverages any available cores (featuring Swift Concurrency). A GPU implementation for applying the activation function is available (featuring Metal) but deactivated because the network is too small to gain performance benefits from it.
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    let section05 = try! AttributedString(markdown:
            """
            **Soap**
            In the early 1990s I read an article about backpropagation in a computer magazine. The author described the then still new method of Rummelhard et al. (1986) and implemented it with Turbo Pascal. I typed the program into my 386 and today forgot what came out. But I stayed true to the overarching theme, and when it became popular again in the 2010s with the availability of cheap, powerful GPUs, the idea of doing something with AI arose. Practical. I had read some papers and books about neural networks and machine learning and was fascinated by the simplicity of the underlying concepts - as far as I understood. With Swift Playgrounds 4 for iPadOS, a kind of lightweight Xcode for app development on the iPad, the barrier became very low that I finally started...
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    
    return ScrollView {
        VStack {
            Text("AIplay").font(.largeTitle)
            HStack { Text(section00).padding() ; Spacer() }
            HStack { Text(section01).padding() ; Spacer() }
            VStack {
                HStack { Text(section02).padding() ; Spacer() }
                VStack {
                    VStack {
                        Image("screenshot")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(minHeight: 0, maxHeight: 512)
                    }
                    VStack {
                        HStack { Text(section03).padding() ; Spacer() }
                    }
                }
            }
            .font(.footnote)
            HStack { Text(section04).padding() ; Spacer() }
            HStack { Text(section05).padding() ; Spacer() }
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(lineWidth: 1))
                .padding()
        }
        Button {
            isPresented.wrappedValue.toggle()
        } label: {
            Text("Close")
        }
    }
    .padding()
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
