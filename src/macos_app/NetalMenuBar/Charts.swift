import SwiftUI

/// Frame time over the last two minutes: the per-window spread as a band, the mean as a line.
///
/// The band is the point. A mean alone hides exactly what matters here - the runtime reports min,
/// mean and max per window, and a flash was always a spike, so a window whose mean is fine while its
/// max is 60 ms should look different from a steady one. It does: the band opens up.
struct FrameTimeChart: View {
    let samples: [MetricsHistory.Sample]
    let ceiling: Double
    /// Drawn as a dashed rule, so "are we holding the refresh rate" is a glance rather than a sum.
    var target: Double = 1000.0 / 60.0

    var body: some View {
        Canvas { context, size in
            guard samples.count > 1 else { return }
            let step = size.width / CGFloat(max(1, MetricsHistory.capacity - 1))
            let x0 = size.width - CGFloat(samples.count - 1) * step
            func point(_ index: Int, _ value: Double) -> CGPoint {
                CGPoint(x: x0 + CGFloat(index) * step,
                        y: size.height * (1 - CGFloat(min(value, ceiling) / ceiling)))
            }

            // Refresh-rate rule.
            if target < ceiling {
                let y = size.height * (1 - CGFloat(target / ceiling))
                var rule = Path()
                rule.move(to: CGPoint(x: 0, y: y)); rule.addLine(to: CGPoint(x: size.width, y: y))
                context.stroke(rule, with: .color(.secondary.opacity(0.35)),
                               style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }

            // min..max band.
            var band = Path()
            band.move(to: point(0, samples[0].maxMs))
            for i in samples.indices.dropFirst() { band.addLine(to: point(i, samples[i].maxMs)) }
            for i in samples.indices.reversed() { band.addLine(to: point(i, samples[i].minMs)) }
            band.closeSubpath()
            context.fill(band, with: .linearGradient(
                Gradient(colors: [.accentColor.opacity(0.32), .accentColor.opacity(0.10)]),
                startPoint: .zero, endPoint: CGPoint(x: 0, y: size.height)))

            // mean line.
            var line = Path()
            line.move(to: point(0, samples[0].meanMs))
            for i in samples.indices.dropFirst() { line.addLine(to: point(i, samples[i].meanMs)) }
            context.stroke(line, with: .color(.accentColor),
                           style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))

            // Anything above the ceiling gets a tick on the top edge: the axis is a percentile, so
            // clipping is expected, and a spike that vanished would be worse than one that clips.
            for i in samples.indices where samples[i].maxMs > ceiling {
                let x = x0 + CGFloat(i) * step
                var tick = Path()
                tick.move(to: CGPoint(x: x, y: 0)); tick.addLine(to: CGPoint(x: x, y: 4))
                context.stroke(tick, with: .color(.orange.opacity(0.9)), lineWidth: 1.5)
            }

            // The newest sample, so the eye lands on "now".
            if let last = samples.indices.last {
                let p = point(last, samples[last].meanMs)
                context.fill(Path(ellipseIn: CGRect(x: p.x - 2.5, y: p.y - 2.5, width: 5, height: 5)),
                             with: .color(.accentColor))
            }
        }
        .drawingGroup()
    }
}

/// A single series as a filled sparkline. Used for SR cost and memory, which have no spread to show.
struct Sparkline: View {
    let values: [Double]
    let ceiling: Double
    var tint: Color = .accentColor

    var body: some View {
        Canvas { context, size in
            guard values.count > 1, ceiling > 0 else { return }
            let step = size.width / CGFloat(max(1, MetricsHistory.capacity - 1))
            let x0 = size.width - CGFloat(values.count - 1) * step
            func point(_ i: Int) -> CGPoint {
                CGPoint(x: x0 + CGFloat(i) * step,
                        y: size.height * (1 - CGFloat(min(values[i], ceiling) / ceiling)))
            }
            var fill = Path()
            fill.move(to: CGPoint(x: x0, y: size.height))
            for i in values.indices { fill.addLine(to: point(i)) }
            fill.addLine(to: CGPoint(x: size.width, y: size.height))
            fill.closeSubpath()
            context.fill(fill, with: .linearGradient(
                Gradient(colors: [tint.opacity(0.28), tint.opacity(0.04)]),
                startPoint: .zero, endPoint: CGPoint(x: 0, y: size.height)))

            var line = Path()
            line.move(to: point(values.startIndex))
            for i in values.indices.dropFirst() { line.addLine(to: point(i)) }
            context.stroke(line, with: .color(tint),
                           style: StrokeStyle(lineWidth: 1.4, lineCap: .round, lineJoin: .round))
        }
        .drawingGroup()
    }
}

/// A titled chart with its current value on the right, sized for a menu-bar panel.
struct ChartCard<Content: View>: View {
    let title: String
    let value: String
    var caption: String?
    var height: CGFloat = 44
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(value)
                    .font(.caption.monospacedDigit().weight(.medium))
                    .contentTransition(.numericText())
            }
            content.frame(height: height)
            if let caption {
                Text(caption).font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }
}
