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
        Button("probe") {
            print((mnist.dataset[.images(.test)] as! [[Float]])[123])
            print((mnist.dataset[.labels(.test)] as! [UInt8])[123])
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
