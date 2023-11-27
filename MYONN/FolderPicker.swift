import SwiftUI

enum FolderPickerError: Error {
    case canceled
}

struct FolderPicker: UIViewControllerRepresentable {
    var completion: ((Result<URL, FolderPickerError>) -> Void)?
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<FolderPicker>) -> UIDocumentPickerViewController {
        let controller = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        controller.delegate = context.coordinator
        controller.allowsMultipleSelection = false
        controller.directoryURL = getAppFolder()
        
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: UIViewControllerRepresentableContext<FolderPicker>) -> Void {
    }
    
    func makeCoordinator() -> FolderPickerCoordinator {
        FolderPickerCoordinator(completion)
    }
}

class FolderPickerCoordinator: NSObject, UINavigationControllerDelegate {
    var completion: ((Result<URL, FolderPickerError>) -> Void)?
    
    init(_  completion: ((Result<URL, FolderPickerError>) -> Void)?) {
        self.completion = completion
    }
}

extension FolderPickerCoordinator: UIDocumentPickerDelegate {
    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) -> Void {
        completion?(.failure(.canceled))
    }
    
    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) -> Void {
        completion?(.success(urls[0]))
    }
}
