import SwiftUI

struct MYONNView: View {
    @State private var folderPickerShow = getAppFolder() == nil
	
	@StateObject private var dataset = MNISTDatset(in: getAppFolder())
    
    var body: some View {
        NetworkView()
            .sheet(isPresented: $folderPickerShow) {
                FolderPicker { result in
                    switch result {
                    case .success(let folder):
                        folder.accessSecurityScopedResource { folder in
                            setAppFolder(url: folder)
                        }
                        dataset.load(from: getAppFolder()!)
                    default: // .failure(let error)
                        break
                    }
                }
            }
    }
}

// NYONN sample configuration
//   usage: GenericFactory.create(NetworkFactory(), defaultConfig)
let defaultConfig: NetworkConfig = (
    layersWithSizes: [784, 100, 10], activationFunctions: [.sigmoid, .sigmoid], learningRate: 0.3
)

// specialized NYONN factory
//   usage: GenericFactory.create(DefaultFactory(), nil)
struct DefaultFactory: AbstractFactory {
    func create(_ config: Never?) -> Network? {
        Network([
            Layer(numberOfInputs: 784, numberOfPUnits: 100, activationFunction: .sigmoid),
            Layer(numberOfInputs: 100, numberOfPUnits: 10, activationFunction: .sigmoid)
        ], alpha: 0.3)
    }
}

extension Matrix where Entry: Comparable {
    func maxValueEntry() -> Entry {
        entries.max(by: { $0 < $1 })! // probably save to force unwrap
    }
    
    func maxValueIndex() -> Int {
        entries.indices.max(by: { entries[$0] < entries[$1] })! // probably save to force unwrap
    }
}
