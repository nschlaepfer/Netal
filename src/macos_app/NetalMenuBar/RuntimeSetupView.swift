import SwiftUI
import AppKit

/// Source-preview setup. A selected folder is a prepared Netal checkout, not an arbitrary DLL.
struct RuntimeSetupView: View {
    @ObservedObject var monitor: RuntimeMonitor
    @State private var selectionNote: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Runtime setup").font(.caption).foregroundStyle(.secondary)
            Text("DLSS needs a prepared Netal runtime folder. Models are supplied separately; installing the app alone does not install SR or FG.")
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(NetalBuildDefaults.runtimeRoot.path).font(.caption2.monospaced())
                .lineLimit(1).truncationMode(.head).help(NetalBuildDefaults.runtimeRoot.path)
            HStack {
                Button("Choose Runtime Folder…", action: choose).controlSize(.small)
                Spacer()
                Link("Setup guide", destination: URL(string: "https://github.com/nschlaepfer/Netal/blob/main/docs/getting-started.md")!)
                    .font(.caption)
            }
            if let selectionNote {
                Text(selectionNote).font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.title = "Choose a prepared Netal runtime folder"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard FileManager.default.isExecutableFile(atPath: url.appendingPathComponent("scripts/netal-native.sh").path),
              FileManager.default.isExecutableFile(atPath: url.appendingPathComponent("scripts/crossover-exact-sr.sh").path) else {
            selectionNote = "That folder does not contain Netal's launch tools. Select the Netal checkout root."
            return
        }
        guard ProcessInfo.processInfo.environment["NETAL_RUNTIME_ROOT"] == nil else {
            selectionNote = "This launch uses NETAL_RUNTIME_ROOT. Remove that override before changing folders here."
            return
        }
        UserDefaults.standard.set(url.path, forKey: "NetalRuntimeRoot")
        UserDefaults.standard.removeObject(forKey: "NetalLogPath")
        monitor.extraLogPaths = [URL(fileURLWithPath: NetalBuildDefaults.crossoverLogPath)]
        monitor.modelBundleFallbacks = [URL(fileURLWithPath: NetalBuildDefaults.srBundlePath)]
        monitor.fgBundleFallbacks = [URL(fileURLWithPath: NetalBuildDefaults.fgBundlePath)]
        monitor.poll()
        selectionNote = "Folder selected. Check the DLSS tab for SR and FG installation status."
    }
}
