import SwiftUI
import PlaygroundTester

struct ContentView: View {
    var body: some View {
        VStack {
            AppIcon()
                .frame(width: /*@START_MENU_TOKEN@*/100/*@END_MENU_TOKEN@*/, height: /*@START_MENU_TOKEN@*/100/*@END_MENU_TOKEN@*/, alignment: /*@START_MENU_TOKEN@*/.center/*@END_MENU_TOKEN@*/)
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
