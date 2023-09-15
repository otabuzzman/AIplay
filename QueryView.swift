import SwiftUI
import PencilKit

struct QueryView: View {
    @Binding var query: [UInt8]
    
    @State private var canvas = PKCanvasView()
    
    var body: some View {
        CanvasView(canvas: $canvas, query: $query)
            .frame(width: 320, height: 320)
    }
}

struct CanvasView {
    @Binding var canvas: PKCanvasView
    var query: Binding<[UInt8]>
}

extension CanvasView: UIViewRepresentable {
    func makeCoordinator() -> CanvasCoordinator {
        CanvasCoordinator(canvas: $canvas, query: query)
    }
    
    func makeUIView(context: Context) -> PKCanvasView {
        canvas.delegate = context.coordinator
        canvas.tool = PKInkingTool(ink: .init(.pencil, color: .darkGray), width: 100)
        canvas.drawingPolicy = .anyInput
        
        return canvas
    }
    
    func updateUIView(_ uiView: PKCanvasView, context: Context) { }
}

class CanvasCoordinator: NSObject {
    private var canvas: Binding<PKCanvasView>
    private var query: Binding<[UInt8]>
    
    private var delay: Task<Void, Never>?
    
    init(canvas: Binding<PKCanvasView>, query: Binding<[UInt8]>) {
        self.canvas = canvas
        self.query = query
    }
}

extension CanvasCoordinator: PKCanvasViewDelegate {
    func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
        if let delay = delay {
            delay.cancel()
        }
        delay = Task {
            do {
                try await Task.sleep(nanoseconds: 1_000_000_000)
            } catch { return }
            query.wrappedValue = await canvasView.makeMNISTImage()
        }
    }
}

extension PKCanvasView {
    func makeMNISTImage() -> [UInt8] {
        guard
            let image = drawing
                .transformed(using: .init(scaleX: 28 / bounds.size.width, y: 28 / bounds.size.height))
                .image(from: .init(x: 0, y: 0, width: 28, height: 28), scale: 1)
                .cgImage
        else { return [] }
        
        var mnist = [UInt8](repeating: 0, count: 784)
        guard
            let context = CGContext(
                data: &mnist,
                width: 28, height: 28,
                bitsPerComponent: 8, bytesPerRow: 28,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return [] }
        
        context.draw(image, in: .init(x: 0, y: 0, width: 28, height: 28))
        
        return mnist
    }
}
