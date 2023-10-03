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

struct AlternateView: View {
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
        ProgressView(value: 0)
        VStack {
            HStack { // mimic list section header look
                Label("NN PREDICTION", systemImage: "wand.and.stars.inverse")
                    .foregroundColor(.secondary).bold(true)
                Spacer()
            }
            .padding(.leading, 12)
            .padding(.bottom, 2)
            HStack {
                Button {
                } label: {
                    Label("Random testset number", systemImage: "sparkle.magnifyingglass")
                }
                Spacer()
                Text("4")
            }
            .padding()
            .background(.white)
            .background(in: RoundedRectangle(cornerRadius: 12))
            Divider()
            HStack {
                VStack {
                    HStack {
                        Text("Input")
                        Spacer()
                        Button {
                        } label: {
                            Image(systemName: "xmark.app")
                        }
                    }
                    Image(systemName: "square.and.pencil")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding()
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(.primary, lineWidth: 1))
                }
                VStack {
                    Text("Image")
                    Image(systemName: "sparkle.magnifyingglass")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding()
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(.primary, lineWidth: 1))
                }
            }
            .padding()
            .background(.white)
            .background(in: RoundedRectangle(cornerRadius: 12))
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        VStack {
            Form {
                Section {
                    HStack {
                        Button {
                        } label: {
                            Label("Reload MNIST", systemImage: "arrow.counterclockwise.icloud")
                        }
                        Spacer()
                        HStack {
                            Circle().foregroundColor(.gray)
                            Circle().foregroundColor(.gray)
                            Circle().foregroundColor(.gray)
                            Circle().foregroundColor(.gray)
                        }
                        .frame(height: 24)
                    }
                } header: {
                    HStack {
                        Label("DATASET", systemImage: "chart.bar").font(.headline)
                        Spacer()
                    }
                }
                Section {
                    HStack {
                        Text("Mini-batch size")
                        Spacer()
                        Text("30")
                    }
                    HStack {
                        Text("Learning rate")
                        Spacer()
                        Text("0.3")
                    }
                } header: {
                    HStack {
                        Label("NN CONFIGURATION", systemImage: "gearshape").font(.headline)
                        Spacer()
                    }
                }
                Section {
                    HStack {
                        Text("Epochs trained so far")
                        Spacer()
                        Text("0")
                    }
                    HStack {
                        Text("Duration of last epoch")
                        Spacer()
                        Text(DateComponentsFormatter().string(from: 1920)!)
                    }
                } header: {
                    HStack {
                        Label("NN TRAINING", systemImage: "dumbbell").font(.headline)
                        Spacer()
                    }
                }
                Section {
                    HStack {
                        Button {
                        } label: {
                            Label("Train next mini-batch", systemImage: "figure.strengthtraining.traditional")
                        }
                        Spacer()
                    } 
                    HStack {
                        Button {
                        } label: {
                            Label("Train another epoch", systemImage: "figure.strengthtraining.traditional")
                        }
                        Spacer()
                    }
                    HStack {
                        Button {
                        } label: {
                            Label("NN Performance", systemImage: "sparkle.magnifyingglass")
                        }
                        Spacer()
                        Text(String(format: "%.4f", 0.9442))
                    }
                    HStack {
                        Button {
                        } label: {
                            Label("Import model from Files", systemImage: "square.and.arrow.up")
                        }
                        Spacer()
                    }
                    HStack {
                        Button {
                        } label: {
                            Label("Export model to Files", systemImage: "square.and.arrow.down")
                        }
                        Spacer()
                    }
                } footer: {
                    HStack {
                        Spacer()
                        Button("Reset") {
                        }
                        .foregroundColor(.red)
                        Spacer()
                    }
                    .padding()
                }
            }
        }
    }
}

func appInfo(isPresented: Binding<Bool>) -> some View {
    let description = try! AttributedString(markdown:
            """
            """, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
    return infoView(isPresented: isPresented, Text(description))
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
