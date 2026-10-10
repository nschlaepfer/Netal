import Foundation

/// A rolling window of pacing samples, for the graphs.
///
/// The runtime emits one sample per 60 frames - about a second - and the monitor polls at the same
/// cadence, so this fills at roughly one point per second. Capacity is two minutes of that, which
/// is long enough to show a spike settling and short enough that the graph stays readable.
struct MetricsHistory {
    struct Sample: Equatable {
        var frame: UInt64
        var fps: Double
        var meanMs: Double
        var minMs: Double
        var maxMs: Double
        var gpuMs: Double?
        var rssMB: Double?
    }

    static let capacity = 120
    private(set) var samples: [Sample] = []

    /// Ignores a repeat of the frame we already hold: the monitor re-reads the tail of the log after
    /// a truncation, and a duplicated point would show as a flat step that never happened.
    mutating func append(_ pacing: Pacing) {
        if let last = samples.last, last.frame == pacing.frame { return }
        samples.append(Sample(frame: pacing.frame, fps: pacing.fps, meanMs: pacing.meanMs,
                              minMs: pacing.minMs, maxMs: pacing.maxMs,
                              gpuMs: pacing.gpuMs, rssMB: pacing.rssMB))
        if samples.count > Self.capacity { samples.removeFirst(samples.count - Self.capacity) }
    }

    mutating func reset() { samples.removeAll(keepingCapacity: true) }

    var isEmpty: Bool { samples.isEmpty }

    /// Upper bound for the frame-time axis.
    ///
    /// Deliberately NOT the worst sample. A session with one 1.6-second stall - which happens, and
    /// is in the logs - would set the ceiling so high that every normal frame collapses onto the
    /// baseline and the graph says nothing. The 90th percentile keeps the axis on the data you are
    /// actually reading, and the handful of samples above it clip at the top, which is the honest
    /// rendering of "off the chart". Rounded to 5 ms so the axis does not jitter per sample.
    var frameTimeCeiling: Double {
        let sorted = samples.map(\.maxMs).sorted()
        guard !sorted.isEmpty else { return 20 }
        let index = min(sorted.count - 1, Int(Double(sorted.count) * 0.9))
        return max(20, (sorted[index] * 1.25 / 5).rounded(.up) * 5)
    }

    var memoryCeiling: Double {
        let peak = samples.compactMap(\.rssMB).max() ?? 0
        return max(512, (peak * 1.1 / 256).rounded(.up) * 256)
    }
}
