import Foundation
import UserNotifications

/// Permission belongs to the user. Denial leaves all status available in the panel;
/// never route around it through another application's notification identity.
@MainActor
enum NetalNotifier {
    private static var authorized = false
    private static var asked = false

    static func requestAuthorization() {
        guard !asked else { return }
        asked = true
        Task { @MainActor in
            authorized = (try? await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound])) ?? false
        }
    }

    static func runSelfTestIfRequested() -> Bool {
        guard CommandLine.arguments.contains("--notify-test") else { return false }
        let report = FileManager.default.temporaryDirectory.appendingPathComponent("netal-notify-test.txt")
        Task { @MainActor in
            do {
                let centre = UNUserNotificationCenter.current()
                let granted = try await centre.requestAuthorization(options: [.alert, .sound])
                guard granted else {
                    try? "authorization=denied delivery=skipped".write(to: report, atomically: true, encoding: .utf8)
                    exit(0)
                }
                let content = UNMutableNotificationContent()
                content.title = "Netal"
                content.body = "Notifications are working."
                try await centre.add(UNNotificationRequest(identifier: "netal.selftest", content: content, trigger: nil))
                try? "authorization=granted delivery=queued result=pass".write(to: report, atomically: true, encoding: .utf8)
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                exit(0)
            } catch {
                try? "delivery=failed detail=\(error.localizedDescription)".write(to: report, atomically: true, encoding: .utf8)
                exit(1)
            }
        }
        return true
    }

    static func post(_ notification: NetalNotification) {
        guard authorized else { return }
        let content = UNMutableNotificationContent()
        content.title = notification.title
        content.body = notification.body
        switch notification.kind {
        case .failed, .flash: content.sound = .default
        default: break
        }
        let request = UNNotificationRequest(identifier: notification.identifier, content: content, trigger: nil)
        Task { try? await UNUserNotificationCenter.current().add(request) }
    }
}
