import SwiftUI
import PencilKit

struct CanvasView {
    @Binding var canvas: PKCanvasView
    @Binding var mNISTImage: MNISTImage
}

extension CanvasView: UIViewRepresentable {
    func makeCoordinator() -> CanvasCoordinator {
        CanvasCoordinator(canvas: $canvas, mNISTImage: $mNISTImage)
    }
    
    func makeUIView(context: Context) -> PKCanvasView {
        canvas.delegate = context.coordinator
        let style = PKInk(.pencil, color: .gray)
        let width = style.inkType.validWidthRange.upperBound
        canvas.tool = PKInkingTool(ink: style, width: width)
        canvas.drawingPolicy = .anyInput
        
        // https://developer.apple.com/forums/thread/120594
        canvas.backgroundColor = .clear
        canvas.isOpaque = false
        
        return canvas
    }

    func updateUIView(_ uiView: PKCanvasView, context: Context) { }
}

class CanvasCoordinator: NSObject {
    private var canvas: Binding<PKCanvasView>
    private var mNISTImage: Binding<MNISTImage>
    
    private var delay: Task<Void, Never>?
    
    init(canvas: Binding<PKCanvasView>, mNISTImage: Binding<MNISTImage>) {
        self.canvas = canvas
        self.mNISTImage = mNISTImage
    }
}

extension CanvasCoordinator: PKCanvasViewDelegate {
    func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
        if canvasView.drawing.bounds.isEmpty {
            return
        }
        if let delay = delay {
            delay.cancel()
        }
        delay = Task {
            do {
                try await Task.sleep(nanoseconds: 1_000_000_000)
            } catch { return }
            mNISTImage.wrappedValue = await canvasView.makeMNISTImage()
        }
    }
}

extension PKCanvasView {
    func makeMNISTImage() -> MNISTImage {
        guard
            let canvasImage = drawing
                .transformed(using: .init(scaleX: 28 / bounds.size.width, y: 28 / bounds.size.height))
                .image(from: .init(x: 0, y: 0, width: 28, height: 28), scale: 32)
                .cgImage
        else { return MNISTImage() }
        
        var mNISTImage = MNISTImage(repeating: 0, count: 784)
        guard
            let mNISTContext = CGContext(
                data: &mNISTImage,
                width: 28, height: 28,
                bitsPerComponent: 8, bytesPerRow: 28,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return MNISTImage() }
        
        mNISTContext.draw(canvasImage, in: .init(x: 0, y: 0, width: 28, height: 28))
        
        return mNISTImage
    }
}

extension Image {
    init?(mNISTImage: MNISTImage) {
        guard
            mNISTImage.count == 784
        else { return nil }
        guard
            let context = CGContext(
                data: nil,
                width: 28, height: 28,
                bitsPerComponent: 8, bytesPerRow: 28,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue + CGImageAlphaInfo.none.rawValue)
        else { return nil }
        context.data?.copyMemory(from: mNISTImage, byteCount: 784)
        guard
            let cgImage = context.makeImage()
        else { return nil }
        let uiImage = UIImage(cgImage: cgImage)
        self.init(uiImage: uiImage)
    }
}
