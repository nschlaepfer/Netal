import Foundation

/// Decides what is worth telling someone about, and — more of the work — what is not.
///
/// A notification that fires while nothing has changed is noise, and noise gets the whole app
/// muted. So this is driven entirely by *transitions*: a run that starts upscaling and keeps
/// upscaling produces exactly one notification, however many frames it serves.
///
/// Deliberately free of UserNotifications so the rules can be tested. Delivery lives in the app.
struct NetalNotification: Equatable {
    enum Kind: String {
        case engaged        // Netal started serving frames
        case stopped        // it was serving, and is not any more
        case fallback       // frames are going to MetalFX instead
        case flash          // a blown frame was detected
        case modelMissing   // a game is attached but there is no model to run
        case failed         // the runtime reported a hard failure
    }
    var kind: Kind
    var title: String
    var body: String
    /// Collapses repeats in Notification Center rather than stacking them.
    var identifier: String
}

struct NotificationPolicy {
    /// What has already been said about the current session, so nothing is said twice.
    private var announced: Set<String> = []
    /// Which session those statements belong to. A new session forgets all of them.
    private var session: String?

    /// Health counters only warrant one notification per session each, however far they climb —
    /// the panel carries the running totals, and a notification per fallback would be unusable.
    private var lastFallbacks: UInt64 = 0
    private var lastFlashes: UInt64 = 0

    mutating func reset(session id: String) {
        if session != id {
            session = id
            announced.removeAll()
            lastFallbacks = 0
            lastFlashes = 0
        }
    }

    /// Nil when nothing should be said.
    mutating func evaluate(previous: RuntimeState, status: RuntimeStatus,
                           game: String?, modelReady: Bool) -> [NetalNotification] {
        var out: [NetalNotification] = []
        let name = game ?? "a game"

        func once(_ key: String, _ make: () -> NetalNotification) {
            guard !announced.contains(key) else { return }
            announced.insert(key)
            out.append(make())
        }

        switch status.state {
        case .active:
            // The one people want: it is on, and here is what it is doing.
            if case .active = previous {} else {
                once("engaged") {
                    let where_ = status.outputResolution.map { " at \($0)" } ?? ""
                    return NetalNotification(kind: .engaged,
                                             title: "Netal is upscaling \(name)",
                                             body: "DLSS super resolution engaged\(where_).",
                                             identifier: "netal.engaged")
                }
            }
        case .failed(let reason):
            once("failed") {
                NetalNotification(kind: .failed, title: "Netal stopped",
                                  body: reason, identifier: "netal.failed")
            }
        case .paused:
            // A pause is something the user just did on purpose. Telling them about it is the
            // definition of noise.
            break
        case .armed, .injected:
            // Attached with no model is a dead end the user has to act on, so it is worth saying —
            // but only once, and only when a game is actually there to run.
            if !modelReady {
                once("model") {
                    NetalNotification(kind: .modelMissing, title: "Netal has no DLSS model",
                                      body: "\(name) is attached but there is nothing to run. "
                                          + "Open Netal ▸ DLSS to supply one.",
                                      identifier: "netal.model")
                }
            }
            // Losing a running session is worth one line; gaining one is covered above.
            if case .active = previous {
                once("stopped") {
                    NetalNotification(kind: .stopped, title: "Netal stopped upscaling",
                                      body: "\(name) is no longer rendering through Netal.",
                                      identifier: "netal.stopped")
                }
            }
        case .noRuntime:
            break
        }

        if status.fallbacks > lastFallbacks, lastFallbacks == 0, status.fallbacks > 0 {
            once("fallback") {
                NetalNotification(kind: .fallback, title: "Netal fell back to MetalFX",
                                  body: "\(status.fallbacks) frame(s) were not upscaled by Netal. "
                                      + "Open Netal for the reason.",
                                  identifier: "netal.fallback")
            }
        }
        lastFallbacks = status.fallbacks

        if status.flashes > lastFlashes, lastFlashes == 0, status.flashes > 0 {
            once("flash") {
                NetalNotification(kind: .flash, title: "Netal detected a blown frame",
                                  body: "The output was not a picture on \(status.flashes) frame(s). "
                                      + "This should not happen — the log has the detail.",
                                  identifier: "netal.flash")
            }
        }
        lastFlashes = status.flashes

        return out
    }
}
