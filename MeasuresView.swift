import SwiftUI
import Charts

struct MeasuresView: View {
    var measures: [Measures]

    var accuracies: [Accuracy] {
        var accuracies = [Accuracy]()
        measures.enumerated().forEach { index, measure in
            let epoch = (index + 1).description
            accuracies.append(Accuracy(epoch: epoch, value: measure.trainingAccuracy, mode: .training))
            accuracies.append(Accuracy(epoch: epoch, value: measure.validationAccuracy, mode: .validation))
        }
        return accuracies
    }
    
    var losses: [(Int, Float)] {
        var losses = [(Int, Float)]()
        measures.enumerated().forEach { epoch, measure in
            if let values = measure.trainingLoss {
                values.enumerated().forEach { index, value in
                    if index % 200 == 0 {
                        losses.append((epoch * values.count + index, value))
                    }
                }
            }
        }
        return losses
    }
    
    var body: some View {
        Chart {
            ForEach(accuracies, id: \.self) { accuracy in
                BarMark(
                    x: .value("epoch", accuracy.epoch),
                    y: .value("accuracy", accuracy.value))
                .position(by: .value("mode", accuracy.mode))
                .foregroundStyle(by: .value("mode", accuracy.mode))
            }
        }
        .chartForegroundStyleScale(domain: Accuracy.Mode.allCases, range: [Color.gray, Color.accentColor])
        .chartXAxisLabel("Epochs", alignment: .center)
        .chartYAxisLabel("Accuracy")
        .chartYAxis {
            AxisMarks(format: Decimal.FormatStyle.Percent.percent.scale(100), values: [0, 0.5, 1.0])
        }
        
        Chart {
            ForEach(losses, id: \.0) { sample in
                LineMark(x: .value("sample", sample.0), y: .value("value", sample.1))
                    .interpolationMethod(.monotone)
                    .foregroundStyle(by: .value("mode", Accuracy.Mode.training))
            }
        }
        .chartForegroundStyleScale(domain: Accuracy.Mode.allCases, range: [Color.gray, Color.accentColor])
        .chartXAxisLabel("Epochs", alignment: .center)
        .chartXAxis {
            AxisMarks(format: IntegerFormatStyle.number.scale(0.001), values: stride(from: 0, to: measures.count, by: 1).map { $0 * 1000 })
        }
        .chartYAxisLabel("Loss")
    }
}

struct Accuracy: Hashable {
    enum Mode: String, CaseIterable {
        case training = "Training"
        case validation = "Validation"
    }
    
    let epoch: String
    let value: Float
    let mode: Mode
}

extension Accuracy.Mode: Plottable {
    var primitivePlottable: String {
        rawValue
    }
}
