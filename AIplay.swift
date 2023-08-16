import SwiftUI
import PlaygroundTester

struct ContentView: View {
    var body: some View {
        VStack {
            AppIcon()
                .frame(width: 100, height: 100, alignment: .center)
            MYONN()
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

struct GenericFactory {
    static func create<Config, Output, Factory: AbstractFactory>(_ object: Factory,_ config: Config) -> Output? where Factory.Config == Config, Factory.Output == Output {
        object.create(config)
    }
}

protocol AbstractFactory {
    associatedtype Config
    associatedtype Output
    func create(_ config: Config) -> Output?
}
