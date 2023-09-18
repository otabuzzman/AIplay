import SwiftUI
import PencilKit

struct QueryView: View {
    var viewModel: NetworkViewModel

    @State private var canvas = PKCanvasView()
    @State private var queryInput: [UInt8] = []
    @State private var queryResult: Int = -1
    @State private var queryTarget: Int = -1

    var body: some View {
        HStack(alignment: .top) {
            Group {
                ZStack {
                    Image(systemName: "square.and.pencil")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .foregroundColor(.gray)
                        .brightness(0.42)
                        .padding(4)
                    CanvasView(canvas: $canvas, mNISTImage: $queryInput)
                        .onChange(of: queryInput) { input in
                            queryResult = viewModel.query(sample: queryInput)
                        }
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(.gray, lineWidth: 2)
                }
                Image(mNISTImage: queryInput)?
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            .frame(width: 140, height: 140)
            Button {
                canvas.drawing = PKDrawing()
                queryResult = -1
                queryTarget = -1
            } label: {
                Image(systemName: "clear")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            .frame(width: 32, height: 32)
            .disabled(canvas.drawing.bounds.isEmpty)
        }
        HStack(alignment: .top) {
            VStack {
                Group {
                    HStack {
                        ForEach(0..<5, id: \.self) {
                            ResultButton(digit: $0, result: $queryResult, target: $queryTarget)
                                .frame(width: 51)
                        }
                    }
                    HStack {
                        ForEach(5..<10, id: \.self) {
                            ResultButton(digit: $0, result: $queryResult, target: $queryTarget)
                                .frame(width: 51)
                        }
                    }
                }
                .disabled(canvas.drawing.bounds.isEmpty)
            }
            Button {
                viewModel.train(sample: queryInput, target: UInt8(queryTarget))
            } label: {
                ZStack {
                    Image(systemName: "square")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                    Image(systemName: "figure.strengthtraining.traditional")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 24, height: 24)
                }
            }
            .frame(width: 32, height: 32)
            .disabled(0 > queryTarget)
        }
    }
}

struct ResultButton: View {
    var digit: Int
    @Binding var result: Int
    @Binding var target: Int

    var body: some View {
        Button("\(digit)") {
            target = digit
        }
        .padding()
        .overlay {
            RoundedRectangle(cornerRadius: 4)
                .stroke(target == digit ? .green :
                            result == digit ? Color.accentColor : .gray, lineWidth: 2)
        }
        .tint(.gray)
        .font(.system(size: 22))
    }
}

struct CanvasView {
    @Binding var canvas: PKCanvasView
    @Binding var mNISTImage: [UInt8]
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
    private var mNISTImage: Binding<[UInt8]>
    
    private var delay: Task<Void, Never>?
    
    init(canvas: Binding<PKCanvasView>, mNISTImage: Binding<[UInt8]>) {
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
    func makeMNISTImage() -> [UInt8] {
        guard
            let canvasImage = drawing
                .transformed(using: .init(scaleX: 28 / bounds.size.width, y: 28 / bounds.size.height))
                .image(from: .init(x: 0, y: 0, width: 28, height: 28), scale: 32)
                .cgImage
        else { return [] }
        
        var mNISTImage = [UInt8](repeating: 0, count: 784)
        guard
            let mNISTContext = CGContext(
                data: &mNISTImage,
                width: 28, height: 28,
                bitsPerComponent: 8, bytesPerRow: 28,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return [] }
        
        mNISTContext.draw(canvasImage, in: .init(x: 0, y: 0, width: 28, height: 28))
        
        return mNISTImage
    }
}

extension Image {
    init?(mNISTImage: [UInt8]) {
        guard
            mNISTImage.count == 784
        else { return nil }
        var data = mNISTImage
        guard
            let context = CGContext(
                data: &data,
                width: 28, height: 28,
                bitsPerComponent: 8, bytesPerRow: 28,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue + CGImageAlphaInfo.none.rawValue),
            let cgImage = context.makeImage()
        else { return nil }
        let uiImage = UIImage(cgImage: cgImage)
        self.init(uiImage: uiImage)
    }
}
