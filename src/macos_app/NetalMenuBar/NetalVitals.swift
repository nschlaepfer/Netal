import Foundation

/// What the character's eyes are saying.
///
/// The orb is the only expressive thing in the app, so it earns its place by carrying live state
/// rather than being decoration. Two channels, deliberately — more than two and it stops reading as
/// a glance and starts needing a legend:
///
///   colour  where the frame time sits on a continuous green-amber-red scale. It is a spectrum
///           rather than four buckets because frame time is continuous: 17 ms and 24 ms are both
///           "fair", and a reading that shows them identically is hiding the thing being watched.
///           The anchor points are the ones the rest of the product already uses: 16.7 ms is the
///           60 Hz line on the graph, and 25 ms is the spike threshold the runtime counts against.
///           Nothing new to learn - the buckets still exist for text, the colour just moves between
///           them smoothly.
///   period  the pulse tracks the actual frame cadence, so a struggling session visibly slows down.
///           This is a readout, not a metaphor: the eyes beat at the rate frames are arriving.
///
/// Colour never carries anything alone — the panel shows the same numbers a few points above, and
/// the state dot shows the state — so this is reinforcement, not the only signal.
struct NetalVitals: Equatable {
    enum Condition: Equatable { case idle, good, fair, poor, failed }

    var condition: Condition = .idle
    /// The reading the colour is drawn from. Nil when there is no frame time to show.
    var meanMs: Double?
    /// Seconds for one full pulse. Nil when nothing should move.
    var period: Double?
    /// 0 when dark, 1 at full glow.
    var intensity: Double = 0

    /// The 60 Hz line from the frame-time graph, and the spike threshold the runtime counts.
    static let comfortable = 1000.0 / 60.0
    static let spike = 25.0

    /// (frame time, hue) anchors for the eye colour. Between them the hue is interpolated, so the
    /// colour moves continuously with the reading; outside them it holds at the end.
    static let colourStops: [(ms: Double, hue: Double)] = [
        (1000.0 / 120.0, 0.42),   // 120 Hz - green, with enough cyan to look calm rather than lurid
        (comfortable,    0.30),   // the 60 Hz line - green
        (spike,          0.12),   // the spike threshold - amber
        (40.0,           0.01),   // 25 fps - red
    ]

    /// Hue on that scale, 0...1 as SwiftUI and AppKit both take it.
    var tintHue: Double {
        if condition == .failed { return 0 }
        guard let meanMs else { return Self.colourStops[1].hue }
        let stops = Self.colourStops
        if meanMs <= stops[0].ms { return stops[0].hue }
        for (lower, upper) in zip(stops, stops.dropFirst()) where meanMs < upper.ms {
            let t = (meanMs - lower.ms) / (upper.ms - lower.ms)
            return lower.hue + (upper.hue - lower.hue) * t
        }
        return stops[stops.count - 1].hue
    }

    /// Soft enough not to look like a warning light when things are fine.
    var tintSaturation: Double { condition == .failed ? 0.80 : 0.62 }

    static func from(state: RuntimeState, pacing: Pacing?) -> NetalVitals {
        switch state {
        case .failed:
            // Steady and unmistakable. A pulsing error looks like activity.
            return NetalVitals(condition: .failed, meanMs: nil, period: nil, intensity: 0.9)
        case .active:
            break
        case .paused, .armed, .injected, .noRuntime:
            return NetalVitals(condition: .idle, meanMs: nil, period: nil, intensity: 0)
        }

        guard let pacing, pacing.meanMs > 0 else {
            return NetalVitals(condition: .good, meanMs: nil, period: 2.0, intensity: 0.5)
        }
        let condition: Condition
        switch pacing.meanMs {
        case ..<comfortable: condition = .good
        case ..<spike:       condition = .fair
        default:             condition = .poor
        }

        // One pulse per frame would be invisible at 60 Hz, so it is scaled to something the eye can
        // follow while keeping the ordering intact: slower frames, slower pulse. Clamped at both
        // ends so a stall does not stop it dead and a very fast game does not strobe.
        // 16.7 ms -> 1.2 s, and proportionally slower from there: 33 ms is twice as slow, 50 ms
        // three times. Clamped so a stall does not stop it dead and a fast game does not strobe.
        let period = min(3.0, max(0.9, 1.2 * (pacing.meanMs / comfortable)))
        return NetalVitals(condition: condition, meanMs: pacing.meanMs, period: period,
                           intensity: 0.85)
    }
}
