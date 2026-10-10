// Public-API, selected-window presentation. This is a separate experimental backend;
// it neither loads DLSS weights nor supplies the game's depth or motion-vector buffers.
import AppKit
import Combine
import ScreenCaptureKit
import Metal
import QuartzCore
import CoreMedia
import CoreVideo
import Carbon

enum CaptureState: String { case idle, preparing, running, stopping, failed }

enum CapturePresentationMode: String, CaseIterable, Identifiable {
    case window, fillDisplay
    var id: String { rawValue }
    var title: String {
        switch self {
        case .window: return "Match game window"
        case .fillDisplay: return "Fill display · controller"
        }
    }
}

struct CaptureGameWindow: Identifiable, Equatable {
    let id: UInt32
    let title: String
    let applicationName: String
    let bundleIdentifier: String
    let processID: pid_t
    let frame: CGRect
}

struct CapturePresentationSample {
    let time: Double
    let generated: Bool
    let spatiallyUpscaled: Bool
}

enum CaptureWindowLevel {
    static func output(above source: Int) -> Int? {
        // Stay below system popup menus. Games using a higher exclusive level
        // must switch to a normal window before click-through output is safe.
        guard source < NSWindow.Level.popUpMenu.rawValue - 1 else { return nil }
        return max(NSWindow.Level.floating.rawValue, source + 1)
    }
}

/// Keep an already-started pair intact while coalescing captures that have not
/// started presentation. At most one trailing source and one latest pair remain.
struct CapturePresentationQueue<Frame> {
    private var pending: (source: Frame, midpoint: Frame?, delay: Double)?
    private var trailing: (frame: Frame, due: Double)?

    mutating func enqueue(source: Frame, midpoint: Frame?, sourceDelay: Double) {
        pending = (source, midpoint, sourceDelay)
    }

    mutating func take(at time: Double) -> Frame? {
        if let trailing {
            guard time + 0.0001 >= trailing.due else { return nil }
            self.trailing = nil
            return trailing.frame
        }
        guard let pending else { return nil }
        self.pending = nil
        if let midpoint = pending.midpoint {
            trailing = (pending.source, time + pending.delay)
            return midpoint
        }
        return pending.source
    }

    mutating func reset() { pending = nil; trailing = nil }
}

/// No capture, permission request, timer, or window is created by initializing this object.
@MainActor
final class GameCaptureEngine: ObservableObject {
    static let shared = GameCaptureEngine()
    static let stopShortcut = "⌃⌥⌘N"

    @Published private(set) var state: CaptureState = .idle
    @Published private(set) var windows: [CaptureGameWindow] = []
    @Published var selectedWindowID: UInt32?
    @Published var scale: Double = 1
    @Published private(set) var superResolution = true
    @Published var frameGeneration = false
    @Published var presentationMode: CapturePresentationMode = .window
    @Published private(set) var capturedFrameCount: UInt64 = 0
    @Published private(set) var presentedFrameCount: UInt64 = 0
    @Published private(set) var generatedFrameCount: UInt64 = 0
    @Published private(set) var spatiallyUpscaledFrameCount: UInt64 = 0
    @Published private(set) var droppedFrameCount: UInt64 = 0
    @Published private(set) var capturedFPS: Double = 0
    @Published private(set) var presentedFPS: Double = 0
    @Published private(set) var processingMilliseconds: Double = 0
    @Published private(set) var errorMessage: String?
    @Published private(set) var statusMessage = "Select an open game window to begin."
    @Published private(set) var sourceResolution = "—"
    @Published private(set) var outputResolution = "—"
    @Published private(set) var interpolationStatus: String?
    @Published private(set) var isRefreshing = false
    private(set) var presentationSamples: [CapturePresentationSample] = []
    private(set) var displayTickSamples: [Double] = []
    private(set) var presentationUsesTransactions = false
    private(set) var submittedPresentationCount: UInt64 = 0
    private(set) var sourceWindowLevel = 0
    private(set) var outputWindowLevel = NSWindow.Level.floating.rawValue
    var requestedDisplayRefreshRate: Int { displayRefreshRate }

    var isRunning: Bool { state == .running }
    var isBusy: Bool { state == .preparing || state == .running || state == .stopping }
    var permissionGranted: Bool { CGPreflightScreenCaptureAccess() }
    var frameGenerationAvailable: Bool { MTLCreateSystemDefaultDevice() != nil }
    var frameGenerationUnavailableReason: String? {
        frameGenerationAvailable ? nil : "A Metal-capable Mac is required."
    }

    private var stream: SCStream?
    private var sink: CaptureStreamSink?
    private var overlay: CaptureOverlayPanel?
    private var renderer: CaptureWindowRenderer?
    private var hotkey: CaptureStopHotkey?
    private var timer: Timer?
    private var activationObserver: NSObjectProtocol?
    private var sleepObserver: NSObjectProtocol?
    private var startTask: Task<Void, Never>?
    private var realtimeActivity: NSObjectProtocol?
    private var generation = UUID()
    private var selectedProcessID: pid_t?
    private var activeMode: CapturePresentationMode = .window
    private var sourceFrame = CGRect.zero
    private var backingScale: CGFloat = 1
    private var capturePixelScale: CGFloat = 1
    private var activeScale: Double = 1
    private var outputSize = CGSize.zero
    private var geometryRevision: UInt64 = 0
    private var updatingGeometry = false
    private var lastInputUptime: Double = 0
    private var previousRateUptime: Double = 0
    private var previousCaptureCount: UInt64 = 0
    private var previousPresentCount: UInt64 = 0
    private var sourceCadence: Double = 1.0 / 60.0
    private var lastSourceTimestamp: Double?
    private var sourceVisible = false
    private var sourceOnScreen = false
    private var pendingStartFrames: (CaptureProcessedFrames, UInt64, Double)?
    private var displayRefreshRate: Int = 60
    private var activeDisplayID: UInt32?
    private var activeDisplayVisibleFrame = CGRect.zero
    private var lastGeometryPoll: Double = 0
    private var missingWindowSince: Double?

    init() {}

    /// Explicit user action; enumeration is never performed at application launch.
    func refreshWindows() {
        guard !isRefreshing else { return }
        errorMessage = nil
        guard obtainPermission() else { return }
        isRefreshing = true
        Task { [weak self] in
            guard let self else { return }
            defer { self.isRefreshing = false }
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
                self.windows = Self.windowChoices(content.windows)
                if !self.windows.contains(where: { $0.id == self.selectedWindowID }) {
                    self.selectedWindowID = self.windows.first?.id
                }
                if !self.isBusy {
                    self.statusMessage = self.windows.isEmpty
                        ? "Open a game in a window, then refresh."
                        : "Choose the game window. Only that window is captured; audio is never captured."
                }
            } catch {
                self.errorMessage = "Could not list windows: \(error.localizedDescription)"
            }
        }
    }

    func start(windowID: UInt32, scale: Double, frameGeneration: Bool, superResolution: Bool = true, expectedProcessID: pid_t? = nil,
               activateGame: Bool = true, allowPermissionRequest: Bool = true,
               allowOutputInspection: Bool = false) {
        guard !isBusy else { return }
        errorMessage = nil
        guard obtainPermission(allowRequest: allowPermissionRequest) else { return }
        guard scale.isFinite, (1...2).contains(scale) else {
            fail("Choose an output scale between 1× and 2×.")
            return
        }
        state = .preparing
        realtimeActivity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .latencyCritical], reason: "Enhance the selected game window")
        statusMessage = "Preparing the selected window…"
        selectedWindowID = windowID
        self.scale = scale
        self.frameGeneration = frameGeneration
        self.superResolution = superResolution
        activeScale = scale
        activeMode = presentationMode
        generation = UUID()
        let run = generation
        let expectedPID = expectedProcessID ?? windows.first(where: { $0.id == windowID })?.processID
        capturedFrameCount = 0; presentedFrameCount = 0; generatedFrameCount = 0
        spatiallyUpscaledFrameCount = 0
        presentationSamples.removeAll(keepingCapacity: true)
        displayTickSamples.removeAll(keepingCapacity: true)
        submittedPresentationCount = 0
        droppedFrameCount = 0; capturedFPS = 0; presentedFPS = 0
        previousCaptureCount = 0; previousPresentCount = 0
        previousRateUptime = ProcessInfo.processInfo.systemUptime
        lastInputUptime = 0; lastSourceTimestamp = nil; geometryRevision = 0
        missingWindowSince = nil
        startTask = Task { [weak self] in
            guard let self else { return }
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
                guard self.generation == run, !Task.isCancelled else { return }
                guard let selected = content.windows.first(where: { $0.windowID == windowID }),
                      selected.owningApplication?.processID != getpid(),
                      expectedPID == nil || selected.owningApplication?.processID == expectedPID,
                      selected.frame.width >= 64, selected.frame.height >= 64,
                      selected.isOnScreen, let application = selected.owningApplication else {
                    throw CaptureEngineError.message("That window is no longer available. Open the game in a window and refresh.")
                }
                guard let device = MTLCreateSystemDefaultDevice() else {
                    throw CaptureEngineError.message("This Mac does not provide a Metal device.")
                }
                // Shader compilation can take seconds on a cold launch. The menu and Stop
                // remain responsive while all device work is prepared off the main actor.
                let prepared = try await Task.detached(priority: .userInitiated) {
                    (try CaptureFrameProcessor(device: device), try CaptureRendererResources(device: device))
                }.value
                guard self.generation == run, !Task.isCancelled else { return }
                let processor = prepared.0
                let renderer = CaptureWindowRenderer(resources: prepared.1)
                guard activateGame || NSWorkspace.shared.frontmostApplication?.processIdentifier == application.processID else {
                    throw CaptureEngineError.message("The selected game is no longer foreground. Capture was not started.")
                }
                let hotkey = try CaptureStopHotkey { [weak self] in self?.stop() }
                let filter = SCContentFilter(desktopIndependentWindow: selected)
                self.sourceFrame = selected.frame
                self.sourceOnScreen = true
                self.capturePixelScale = CGFloat(filter.pointPixelScale)
                self.selectedProcessID = application.processID
                let screen = Self.screen(for: selected.frame)
                self.backingScale = screen?.backingScaleFactor ?? self.capturePixelScale
                self.displayRefreshRate = max(30, min(240, screen?.maximumFramesPerSecond ?? 60))
                renderer.setRefreshRate(self.displayRefreshRate)
                self.activeDisplayID = Self.displayID(screen)
                self.activeDisplayVisibleFrame = screen?.visibleFrame ?? .zero
                let dimensions = self.dimensions(for: selected.frame, screen: screen)
                self.outputSize = dimensions.output
                let configuration = self.configuration(width: dimensions.input.width,
                                                       height: dimensions.input.height,
                                                       frameGeneration: frameGeneration)
                let overlay = CaptureOverlayPanel(contentRect: self.presentationRect(for: selected.frame, screen: screen))
                guard let level = CaptureWindowLevel.output(above: selected.windowLayer) else {
                    throw CaptureEngineError.message("Switch the game to windowed or borderless mode before starting enhancement.")
                }
                self.sourceWindowLevel = selected.windowLayer
                self.outputWindowLevel = level
                overlay.level = NSWindow.Level(rawValue: level)
                // Opt-in diagnostics can inspect this selected game's output. Normal
                // sessions remain excluded from other apps' window captures.
                if allowOutputInspection { overlay.sharingType = .readOnly }
                overlay.contentView = renderer.view
                renderer.resize(pixelSize: overlay.frame.size.applying(.init(scaleX: self.backingScale,
                                                                           y: self.backingScale)))
                let sink = CaptureStreamSink(processor: processor,
                    outputSize: dimensions.output,
                    expectedInputSize: dimensions.input,
                    frameGeneration: frameGeneration,
                    onFrames: { [weak self] frames, revision, timestamp in
                        guard let self, self.generation == run else { return }
                        self.receive(frames, revision: revision, timestamp: timestamp)
                    }, onError: { [weak self] error in
                        Task { @MainActor in
                            guard let self, self.generation == run else { return }
                            self.fail(error)
                        }
                    })
                let stream = SCStream(filter: filter, configuration: configuration, delegate: sink)
                try stream.addStreamOutput(sink, type: .screen,
                    sampleHandlerQueue: DispatchQueue(label: "netal.window.capture", qos: .userInteractive))
                self.renderer = renderer; self.overlay = overlay; self.hotkey = hotkey
                self.sink = sink; self.stream = stream
                self.setResolutionLabels(dimensions)
                renderer.onPresented = { [weak self] generated, time, upscaled in
                    Task { @MainActor in
                        guard let self, self.generation == run, self.state == .running else { return }
                        self.presentedFrameCount += 1
                        if generated { self.generatedFrameCount += 1 }
                        if upscaled { self.spatiallyUpscaledFrameCount += 1 }
                        self.presentationSamples.append(.init(time: time, generated: generated, spatiallyUpscaled: upscaled))
                        if self.presentationSamples.count > 4096 { self.presentationSamples.removeFirst() }
                    }
                }
                renderer.onDisplayTick = { [weak self] time, usesTransactions in
                    guard let self, self.generation == run, self.state == .running else { return }
                    self.displayTickSamples.append(time)
                    self.presentationUsesTransactions = usesTransactions
                    if self.displayTickSamples.count > 4096 { self.displayTickSamples.removeFirst() }
                }
                renderer.onSubmitted = { [weak self] in
                    guard let self, self.generation == run, self.state == .running else { return }
                    self.submittedPresentationCount += 1
                }
                renderer.onError = { [weak self] message in
                    guard let self, self.generation == run else { return }
                    self.fail(message)
                }
                try await stream.startCapture()
                guard self.generation == run, !Task.isCancelled else {
                    try? await stream.stopCapture()
                    return
                }
                guard activateGame || NSWorkspace.shared.frontmostApplication?.processIdentifier == application.processID else {
                    throw CaptureEngineError.message("The selected game lost foreground status while capture started.")
                }
                self.state = .running
                self.statusMessage = "Waiting for the game's first frame… · Stop \(Self.stopShortcut)"
                self.installMonitoring(run: run)
                // The output panel itself can never take keyboard focus.
                if activateGame { NSRunningApplication(processIdentifier: application.processID)?.activate(options: []) }
                self.updateVisibility()
                if let first = self.pendingStartFrames {
                    self.pendingStartFrames = nil
                    self.receive(first.0, revision: first.1, timestamp: first.2)
                }
            } catch {
                guard self.generation == run else { return }
                self.fail(error.localizedDescription)
            }
        }
    }

    /// Hides the overlay synchronously, then releases capture in the background.
    /// Late GPU completions and callbacks carry a different generation and are ignored.
    func stop() { finish(error: nil) }

    private func obtainPermission(allowRequest: Bool = true) -> Bool {
        guard !CGPreflightScreenCaptureAccess() else { return true }
        if allowRequest, CGRequestScreenCaptureAccess() { return true }
        errorMessage = "Allow Netal in System Settings → Privacy & Security → Screen & System Audio Recording, then reopen Netal if macOS requests it. Netal captures only your selected game window, without audio."
        if !isBusy { state = .failed }
        statusMessage = "Screen Recording permission is required for Window mode."
        return false
    }

    private func fail(_ message: String) { finish(error: message) }

    private func finish(error: String?) {
        if let realtimeActivity { ProcessInfo.processInfo.endActivity(realtimeActivity) }
        realtimeActivity = nil
        generation = UUID()
        startTask?.cancel(); startTask = nil
        overlay?.orderOut(nil); overlay?.close(); overlay = nil
        renderer?.invalidate(); renderer?.onPresented = nil; renderer = nil
        hotkey = nil
        timer?.invalidate(); timer = nil
        if let activationObserver { NSWorkspace.shared.notificationCenter.removeObserver(activationObserver) }
        if let sleepObserver { NSWorkspace.shared.notificationCenter.removeObserver(sleepObserver) }
        activationObserver = nil; sleepObserver = nil
        if let counts = sink?.snapshot() { capturedFrameCount = counts.captured; droppedFrameCount = counts.dropped }
        sink?.invalidate(); sink = nil
        let oldStream = stream; stream = nil
        selectedProcessID = nil; sourceVisible = false; sourceOnScreen = false; updatingGeometry = false
        pendingStartFrames = nil
        capturedFPS = 0; presentedFPS = 0
        errorMessage = error
        state = error == nil ? .stopping : .failed
        statusMessage = error == nil ? "Stopped. Your game is unchanged." : "Window mode stopped."
        let run = generation
        Task { [weak self] in
            if let oldStream { try? await oldStream.stopCapture() }
            guard let self, self.generation == run else { return }
            if error == nil { self.state = .idle }
        }
    }

    private func receive(_ frames: CaptureProcessedFrames, revision: UInt64, timestamp: Double) {
        if state == .preparing { pendingStartFrames = (frames, revision, timestamp); return }
        guard state == .running, revision == geometryRevision, !updatingGeometry else { return }
        lastInputUptime = ProcessInfo.processInfo.systemUptime
        processingMilliseconds = frames.gpuMilliseconds ?? 0
        interpolationStatus = frames.frameGenerationNote
        if let previous = lastSourceTimestamp {
            let interval = timestamp - previous
            if interval > 0, interval < 0.25 { sourceCadence = interval }
        }
        lastSourceTimestamp = timestamp
        updateVisibility()
        guard sourceVisible, let overlay, let renderer else { return }
        if !overlay.isVisible { overlay.orderFrontRegardless() }
        // Quantize half-steps to display ticks. Small capture timestamp jitter must
        // not accidentally defer a source frame by a whole additional refresh.
        let halfTicks = max(1, (min(0.05, sourceCadence / 2) * Double(displayRefreshRate)).rounded())
        renderer.present(source: frames.source, midpoint: frameGeneration ? frames.midpoint : nil,
                         upscaled: frames.spatiallyUpscaled,
                         sourceDelay: halfTicks / Double(displayRefreshRate))
        let processing = frameGeneration ? "experimental motion interpolation"
            : (frames.spatiallyUpscaled ? "MetalFX spatial SR" : "original-resolution frames")
        statusMessage = "Window mode · \(processing) · Stop \(Self.stopShortcut)"
    }

    private func installMonitoring(run: UUID) {
        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.updateVisibility() }
            }
        sleepObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.willSleepNotification, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.stop() }
            }
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == run else { return }
                self.poll()
            }
        }
        if let timer { RunLoop.main.add(timer, forMode: .common) }
    }

    private func poll() {
        guard state == .running, let selectedWindowID else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if let counts = sink?.snapshot() {
            capturedFrameCount = counts.captured
            droppedFrameCount = counts.dropped
        }
        let elapsed = now - previousRateUptime
        if elapsed >= 1 {
            capturedFPS = Double(capturedFrameCount - previousCaptureCount) / elapsed
            presentedFPS = Double(presentedFrameCount - previousPresentCount) / elapsed
            previousCaptureCount = capturedFrameCount; previousPresentCount = presentedFrameCount
            previousRateUptime = now
        }
        if lastInputUptime > 0, now - lastInputUptime > 0.75 {
            overlay?.orderOut(nil)
            statusMessage = "Waiting for new game frames… · Stop \(Self.stopShortcut)"
        }
        guard now - lastGeometryPoll >= 0.2 else { return }
        lastGeometryPoll = now
        let matches: ([String: Any]) -> Bool = {
            ($0[kCGWindowNumber as String] as? NSNumber)?.uint32Value == selectedWindowID
        }
        let descriptions = CGWindowListCopyWindowInfo(.optionIncludingWindow, selectedWindowID) as? [[String: Any]] ?? []
        let description = descriptions.first(where: matches)
            ?? (CGWindowListCopyWindowInfo(.optionAll, kCGNullWindowID) as? [[String: Any]])?.first(where: matches)
        guard let description else {
            // CrossOver can temporarily omit a window during a display transition.
            // Hide immediately, keep the pinned capture filter, and allow a brief
            // recovery. Never select another window or change the owning process.
            sourceOnScreen = false; sourceVisible = false
            renderer?.discardPendingFrames(); overlay?.orderOut(nil)
            if missingWindowSince == nil { missingWindowSince = now }
            if now - (missingWindowSince ?? now) >= 2 {
                fail("The selected game window closed. Reopen it and choose the new window.")
            } else {
                statusMessage = "Waiting for the game window to return… · Stop \(Self.stopShortcut)"
            }
            return
        }
        missingWindowSince = nil
        guard (description[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == selectedProcessID,
              let bounds = description[kCGWindowBounds as String] as? [String: Any],
              let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary) else {
            fail("The selected game window closed. Reopen it and choose the new window.")
            return
        }
        let onScreen = (description[kCGWindowIsOnscreen as String] as? Bool) ?? false
        sourceOnScreen = onScreen
        let sourceLevel = (description[kCGWindowLayer as String] as? NSNumber)?.intValue ?? 0
        guard let level = CaptureWindowLevel.output(above: sourceLevel) else {
            fail("The game switched to an exclusive window level. Use windowed or borderless mode and start again.")
            return
        }
        sourceWindowLevel = sourceLevel; outputWindowLevel = level
        if overlay?.level.rawValue != level { overlay?.level = NSWindow.Level(rawValue: level) }
        if !onScreen || frame.width < 64 || frame.height < 64 {
            sourceVisible = false; overlay?.orderOut(nil)
            statusMessage = "Game window is hidden or minimized. Restore it to continue."
            return
        }
        let screen = Self.screen(for: frame)
        let newBacking = screen?.backingScaleFactor ?? 1
        let newRefresh = max(30, min(240, screen?.maximumFramesPerSecond ?? 60))
        if frame.size != sourceFrame.size || newBacking != backingScale ||
            Self.displayID(screen) != activeDisplayID || newRefresh != displayRefreshRate ||
            screen?.visibleFrame != activeDisplayVisibleFrame {
            updateGeometry(frame, screen: screen)
        } else if frame.origin != sourceFrame.origin {
            sourceFrame = frame
            overlay?.setFrame(presentationRect(for: frame, screen: screen), display: false)
        }
        updateVisibility()
    }

    private func updateVisibility() {
        guard state == .running else { overlay?.orderOut(nil); return }
        let front = NSWorkspace.shared.frontmostApplication?.processIdentifier
        // Netal may be frontmost while its menu is open, but another app must never be covered.
        sourceVisible = sourceOnScreen && (front == selectedProcessID || front == getpid())
        if !sourceVisible {
            renderer?.discardPendingFrames()
            overlay?.orderOut(nil)
        }
    }

    private func updateGeometry(_ frame: CGRect, screen: NSScreen?) {
        guard !updatingGeometry, let stream else { return }
        updatingGeometry = true
        geometryRevision += 1
        let revision = geometryRevision
        sourceFrame = frame
        backingScale = screen?.backingScaleFactor ?? 1
        capturePixelScale = backingScale
        displayRefreshRate = max(30, min(240, screen?.maximumFramesPerSecond ?? 60))
        renderer?.setRefreshRate(displayRefreshRate)
        activeDisplayID = Self.displayID(screen)
        activeDisplayVisibleFrame = screen?.visibleFrame ?? .zero
        let dimensions = dimensions(for: frame, screen: screen)
        outputSize = dimensions.output
        overlay?.orderOut(nil)
        overlay?.setFrame(presentationRect(for: frame, screen: screen), display: false)
        if let overlay { renderer?.resize(pixelSize: overlay.frame.size.applying(.init(scaleX: backingScale, y: backingScale))) }
        sink?.configure(outputSize: dimensions.output, inputSize: dimensions.input, revision: revision)
        let config = configuration(width: dimensions.input.width, height: dimensions.input.height,
                                   frameGeneration: frameGeneration)
        let run = generation
        Task { [weak self] in
            do {
                try await stream.updateConfiguration(config)
                guard let self, self.generation == run else { return }
                self.updatingGeometry = false
                self.lastSourceTimestamp = nil
                self.setResolutionLabels(dimensions)
            } catch {
                guard let self, self.generation == run else { return }
                self.fail("The game window changed, but capture could not resize: \(error.localizedDescription)")
            }
        }
    }

    private func configuration(width: CGFloat, height: CGFloat, frameGeneration: Bool) -> SCStreamConfiguration {
        let configuration = SCStreamConfiguration()
        configuration.width = Int(width); configuration.height = Int(height)
        configuration.pixelFormat = kCVPixelFormatType_32BGRA
        configuration.colorSpaceName = CGColorSpace.sRGB
        configuration.queueDepth = 3
        configuration.showsCursor = false
        configuration.capturesAudio = false
        configuration.ignoreShadowsSingleWindow = true
        configuration.scalesToFit = true
        configuration.preservesAspectRatio = true
        configuration.captureResolution = .best
        // The presentation path has only the display's refresh budget. In 2× mode source
        // capture is bounded to half that budget; these are capture rates, not game FPS.
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: Int32(frameGeneration ? max(15, displayRefreshRate / 2) : displayRefreshRate))
        return configuration
    }

    private func dimensions(for frame: CGRect, screen: NSScreen?) -> (input: CGSize, output: CGSize) {
        let input = CGSize(width: max(2, Int(frame.width * capturePixelScale) / 2 * 2),
                           height: max(2, Int(frame.height * capturePixelScale) / 2 * 2))
        // Presentation size and SR are independent. With SR off, the renderer may
        // fit the original pixels to the display, but no MetalFX pass is executed.
        guard superResolution else { return (input, input) }
        let target: CGSize
        if activeMode == .fillDisplay, let screen {
            target = screen.visibleFrame.size.applying(.init(scaleX: backingScale, y: backingScale))
        } else {
            target = frame.size.applying(.init(scaleX: backingScale * activeScale, y: backingScale * activeScale))
        }
        // Never intentionally discard source pixels to fabricate an upscaling benefit.
        let ratio = max(1, min(target.width / input.width, target.height / input.height))
        let bounded = max(1, min(ratio, 7680 / max(input.width, input.height)))
        let output = CGSize(width: max(2, Int(input.width * bounded) / 2 * 2),
                            height: max(2, Int(input.height * bounded) / 2 * 2))
        return (input, output)
    }

    private func setResolutionLabels(_ size: (input: CGSize, output: CGSize)) {
        sourceResolution = "\(Int(size.input.width)) × \(Int(size.input.height))"
        outputResolution = "\(Int(size.output.width)) × \(Int(size.output.height))"
    }

    private func presentationRect(for frame: CGRect, screen: NSScreen?) -> CGRect {
        if activeMode == .fillDisplay, let screen { return screen.visibleFrame }
        return Self.appKitRect(frame)
    }

    static func appKitRect(_ frame: CGRect) -> CGRect {
        let top = NSScreen.screens.first?.frame.maxY ?? 0
        return CGRect(x: frame.minX, y: top - frame.maxY, width: frame.width, height: frame.height)
    }

    private static func screen(for frame: CGRect) -> NSScreen? {
        let appFrame = appKitRect(frame)
        return NSScreen.screens.max { left, right in
            let l = left.frame.intersection(appFrame); let r = right.frame.intersection(appFrame)
            return (l.isNull ? 0 : l.width * l.height) < (r.isNull ? 0 : r.width * r.height)
        }
    }

    private static func displayID(_ screen: NSScreen?) -> UInt32? {
        (screen?.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
    }

    static func windowChoices(_ source: [SCWindow]) -> [CaptureGameWindow] {
        source.compactMap { window in
            guard let app = window.owningApplication,
                  app.processID != getpid(), window.windowLayer == 0,
                  window.isOnScreen, window.frame.width >= 128, window.frame.height >= 128 else { return nil }
            return CaptureGameWindow(id: window.windowID, title: window.title ?? app.applicationName,
                applicationName: app.applicationName, bundleIdentifier: app.bundleIdentifier,
                processID: app.processID, frame: window.frame)
        }.sorted {
            ($0.applicationName.localizedLowercase, $0.title.localizedLowercase, $0.id) <
            ($1.applicationName.localizedLowercase, $1.title.localizedLowercase, $1.id)
        }
    }
}

private enum CaptureEngineError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case let .message(text) = self { return text }; return nil }
}

/// One processing job plus one replaceable pending frame. SCK's callback queue never
/// waits for GPU work and replaced IOSurfaces are released immediately.
private final class CaptureStreamSink: NSObject, SCStreamOutput, SCStreamDelegate {
    private struct Job {
        let buffer: CVPixelBuffer
        let timestamp: Double
        let outputSize: CGSize
        let revision: UInt64
    }
    private let lock = NSLock()
    private let worker = DispatchQueue(label: "netal.window.processing", qos: .userInteractive)
    private let processor: CaptureFrameProcessor
    private let frameGeneration: Bool
    private var pending: Job?
    private var busy = false
    private var active = true
    private var outputSize: CGSize
    private var inputSize: CGSize
    private var revision: UInt64 = 0
    private var captured: UInt64 = 0
    private var dropped: UInt64 = 0
    private var lastProcessedTimestamp: Double?
    private var processedRevision: UInt64?
    private let onFrames: @MainActor (CaptureProcessedFrames, UInt64, Double) -> Void
    private let onError: (String) -> Void
    private var pendingDelivery: (CaptureProcessedFrames, UInt64, Double)?
    private var deliveryScheduled = false

    init(processor: CaptureFrameProcessor, outputSize: CGSize, expectedInputSize: CGSize,
         frameGeneration: Bool, onFrames: @MainActor @escaping (CaptureProcessedFrames, UInt64, Double) -> Void,
         onError: @escaping (String) -> Void) {
        self.processor = processor; self.outputSize = outputSize; self.inputSize = expectedInputSize
        self.frameGeneration = frameGeneration; self.onFrames = onFrames; self.onError = onError
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid,
              let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false)
                as? [[SCStreamFrameInfo: Any]],
              let rawStatus = attachments.first?[.status] as? Int,
              let status = SCFrameStatus(rawValue: rawStatus) else { return }
        if status == .stopped { onError("The selected game window stopped sharing."); return }
        guard status == .complete, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMTimeGetSeconds(sampleBuffer.presentationTimeStamp)
        guard timestamp.isFinite else { return }
        lock.lock()
        guard active else { lock.unlock(); return }
        captured += 1
        guard CVPixelBufferGetWidth(buffer) == Int(inputSize.width),
              CVPixelBufferGetHeight(buffer) == Int(inputSize.height) else {
            dropped += 1; lock.unlock(); return
        }
        if pending != nil { dropped += 1 }
        pending = Job(buffer: buffer, timestamp: timestamp, outputSize: outputSize, revision: revision)
        let launch = !busy
        busy = true
        lock.unlock()
        if launch { worker.async { [weak self] in self?.drain() } }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        lock.lock(); let report = active; lock.unlock()
        if report { onError("Window capture ended: \(error.localizedDescription)") }
    }

    func configure(outputSize: CGSize, inputSize: CGSize, revision: UInt64) {
        lock.lock(); defer { lock.unlock() }
        self.outputSize = outputSize; self.inputSize = inputSize; self.revision = revision
        if pending != nil { dropped += 1 }; pending = nil; pendingDelivery = nil
    }

    func invalidate() {
        lock.lock(); active = false; pending = nil; pendingDelivery = nil; lock.unlock()
        worker.async { [processor] in processor.reset() }
    }

    func snapshot() -> (captured: UInt64, dropped: UInt64) {
        lock.lock(); defer { lock.unlock() }; return (captured, dropped)
    }

    private func drain() {
        while true {
            lock.lock()
            guard active, let job = pending else { busy = false; lock.unlock(); return }
            pending = nil
            lock.unlock()
            let hostNow = CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
            if hostNow - job.timestamp > 0.25 {
                lock.lock(); dropped += 1; lock.unlock(); continue
            }
            if processedRevision != job.revision ||
                lastProcessedTimestamp.map({ job.timestamp - $0 > 0.25 || job.timestamp <= $0 }) == true {
                processor.reset()
            }
            processedRevision = job.revision; lastProcessedTimestamp = job.timestamp
            do {
                let result = try autoreleasepool {
                    try processor.process(pixelBuffer: job.buffer, outputWidth: Int(job.outputSize.width),
                                          outputHeight: Int(job.outputSize.height), frameGeneration: frameGeneration)
                }
                lock.lock(); let deliver = active && revision == job.revision; lock.unlock()
                if deliver { queueDelivery(result, revision: job.revision, timestamp: job.timestamp) }
            } catch {
                lock.lock(); let report = active; active = false; pending = nil; busy = false; lock.unlock()
                if report { onError("Window processing failed: \(error.localizedDescription)") }
                return
            }
        }
    }

    private func queueDelivery(_ result: CaptureProcessedFrames, revision: UInt64, timestamp: Double) {
        lock.lock()
        guard active, self.revision == revision else { lock.unlock(); return }
        if pendingDelivery != nil { dropped += 1 }
        pendingDelivery = (result, revision, timestamp)
        let schedule = !deliveryScheduled
        deliveryScheduled = true
        lock.unlock()
        guard schedule else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let delivery = self.active ? self.pendingDelivery : nil
            self.pendingDelivery = nil; self.deliveryScheduled = false
            self.lock.unlock()
            if let delivery {
                let hostNow = CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
                guard hostNow - delivery.2 <= 0.25 else { return }
                self.onFrames(delivery.0, delivery.1, delivery.2)
            }
        }
    }
}

@MainActor
private final class CaptureOverlayPanel: NSPanel {
    init(contentRect: CGRect) {
        super.init(contentRect: contentRect, styleMask: [.borderless, .nonactivatingPanel],
                   backing: .buffered, defer: false)
        title = "Netal Window Output"
        isReleasedWhenClosed = false
        ignoresMouseEvents = true
        acceptsMouseMovedEvents = false
        becomesKeyOnlyIfNeeded = true
        hidesOnDeactivate = false
        hasShadow = false
        isOpaque = true
        backgroundColor = .black
        sharingType = .none
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        animationBehavior = .none
    }
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

private final class CaptureRendererResources: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState
    init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw CaptureEngineError.message("Could not create a presentation queue.") }
        self.queue = queue
        let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        struct Out { float4 position [[position]]; float2 uv; };
        vertex Out captureVertex(uint id [[vertex_id]]) {
            float2 p = float2((id << 1) & 2, id & 2);
            return { float4(p * 2.0 - 1.0, 0.0, 1.0), float2(p.x, 1.0 - p.y) };
        }
        fragment float4 captureFragment(Out in [[stage_in]], texture2d<float> image [[texture(0)]]) {
            constexpr sampler sampleFilter(coord::normalized, address::clamp_to_edge, filter::linear);
            return float4(image.sample(sampleFilter, in.uv).rgb, 1.0);
        }
        """, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "captureVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "captureFragment")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    }
}

@MainActor
private final class CaptureWindowRenderer {
    let view: NSView
    private let layer: CAMetalLayer
    private var worker: CaptureRenderWorker!
    var onPresented: ((Bool, Double, Bool) -> Void)?
    var onDisplayTick: ((Double, Bool) -> Void)?
    var onSubmitted: (() -> Void)?
    var onError: ((String) -> Void)?

    init(resources: CaptureRendererResources) {
        layer = CAMetalLayer()
        layer.device = resources.device
        layer.pixelFormat = .bgra8Unorm
        layer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        layer.framebufferOnly = true
        layer.displaySyncEnabled = true
        layer.maximumDrawableCount = 3
        layer.allowsNextDrawableTimeout = true
        view = NSView(frame: .zero)
        view.layer = layer
        view.wantsLayer = true
        view.autoresizingMask = [.width, .height]
        worker = CaptureRenderWorker(layer: layer, resources: resources,
            onPresented: { [weak self] generated, time, upscaled in self?.onPresented?(generated, time, upscaled) },
            onSubmitted: { [weak self] in self?.onSubmitted?() },
            onDisplayTick: { [weak self] time, transaction in self?.onDisplayTick?(time, transaction) },
            onError: { [weak self] message in self?.onError?(message) })
    }

    deinit { worker?.invalidate() }

    func setRefreshRate(_ rate: Int) { worker.setRefreshRate(rate) }
    func resize(pixelSize: CGSize) {
        // This is a layer-hosting view: AppKit does not size our custom layer.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        layer.frame = view.bounds
        layer.contentsScale = view.bounds.width > 0 ? pixelSize.width / view.bounds.width : 1
        layer.drawableSize = pixelSize
        CATransaction.commit()
        worker.resize(pixelSize)
    }
    func invalidate() { worker.invalidate() }
    func discardPendingFrames() { worker.discardPendingFrames() }
    func present(source: MTLTexture, midpoint: MTLTexture?, upscaled: Bool, sourceDelay: Double) {
        worker.present(source: source, midpoint: midpoint, upscaled: upscaled, sourceDelay: sourceDelay)
    }
}

/// A display tick consumes at most one new frame. Never repeat a texture just to
/// increase a counter; GPU encoding stays off AppKit and capture queues.
private final class CaptureRenderWorker: NSObject, CAMetalDisplayLinkDelegate, @unchecked Sendable {
    private let layer: CAMetalLayer
    private let resources: CaptureRendererResources
    private var thread: Thread?
    private var runLoop: CFRunLoop?
    private var refreshRate = 60
    private let lock = NSLock()
    private struct Frame {
        let texture: MTLTexture
        let generated: Bool
        let upscaled: Bool
    }
    private var frames = CapturePresentationQueue<Frame>()
    private var active = true
    private var size = CGSize(width: 1, height: 1)
    private let available = DispatchSemaphore(value: 2)
    private let onPresented: @MainActor (Bool, Double, Bool) -> Void
    private let onSubmitted: @MainActor () -> Void
    private let onDisplayTick: @MainActor (Double, Bool) -> Void
    private let onError: @MainActor (String) -> Void

    init(layer: CAMetalLayer, resources: CaptureRendererResources,
         onPresented: @MainActor @escaping (Bool, Double, Bool) -> Void,
         onSubmitted: @MainActor @escaping () -> Void,
         onDisplayTick: @MainActor @escaping (Double, Bool) -> Void,
         onError: @MainActor @escaping (String) -> Void) {
        self.layer = layer; self.resources = resources
        self.onPresented = onPresented; self.onError = onError
        self.onSubmitted = onSubmitted
        self.onDisplayTick = onDisplayTick
        super.init()
        let thread = Thread { [weak self] in self?.runDisplayLoop() }
        thread.name = "netal.window.presentation"
        thread.qualityOfService = .userInteractive
        self.thread = thread
        thread.start()
    }

    private func runDisplayLoop() {
        let loop = CFRunLoopGetCurrent()!
        lock.lock()
        guard active else { lock.unlock(); return }
        runLoop = loop
        let fps = Float(refreshRate)
        lock.unlock()
        let link = CAMetalDisplayLink(metalLayer: layer)
        link.delegate = self
        link.preferredFrameLatency = 2
        link.preferredFrameRateRange = CAFrameRateRange(minimum: fps, maximum: fps, preferred: fps)
        link.add(to: .current, forMode: .common)
        // A port keeps the run loop alive while the layer is temporarily hidden.
        let port = Port()
        RunLoop.current.add(port, forMode: .default)
        while true {
            lock.lock(); let running = active; lock.unlock()
            if !running { break }
            autoreleasepool { _ = RunLoop.current.run(mode: .default, before: .distantFuture) }
        }
        link.invalidate()
        port.invalidate()
        lock.lock(); runLoop = nil; lock.unlock()
    }

    func setRefreshRate(_ rate: Int) {
        lock.lock(); refreshRate = max(30, min(240, rate)); lock.unlock()
    }

    func resize(_ pixelSize: CGSize) {
        lock.lock(); size = pixelSize; frames.reset(); lock.unlock()
    }

    func invalidate() {
        lock.lock(); active = false; frames.reset(); let loop = runLoop; lock.unlock()
        if let loop { CFRunLoopStop(loop); CFRunLoopWakeUp(loop) }
    }
    func discardPendingFrames() { lock.lock(); frames.reset(); lock.unlock() }

    func present(source: MTLTexture, midpoint: MTLTexture?, upscaled: Bool, sourceDelay: Double) {
        lock.lock()
        guard active else { lock.unlock(); return }
        frames.enqueue(source: Frame(texture: source, generated: false, upscaled: upscaled),
                       midpoint: midpoint.map { Frame(texture: $0, generated: true, upscaled: upscaled) },
                       sourceDelay: sourceDelay)
        lock.unlock()
    }

    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        lock.lock()
        let fps = Float(refreshRate)
        let running = active
        lock.unlock()
        guard running else { return }
        if link.preferredFrameRateRange.preferred != fps {
            link.preferredFrameRateRange = CAFrameRateRange(minimum: fps, maximum: fps, preferred: fps)
        }
        let time = update.targetPresentationTimestamp
        let transaction = layer.presentsWithTransaction
        DispatchQueue.main.async { [weak self] in self?.onDisplayTick(time, transaction) }
        lock.lock()
        guard active, available.wait(timeout: .now()) == .success else { lock.unlock(); return }
        guard let next = frames.take(at: update.targetPresentationTimestamp) else {
            available.signal(); lock.unlock(); return
        }
        let size = self.size
        let drawable = update.drawable
        lock.unlock()
        // Finish submission within the display callback, on its dedicated thread.
        // Dispatching this work after the callback can miss the drawable deadline.
        autoreleasepool {
            render(next.texture, generated: next.generated, upscaled: next.upscaled,
                   size: size, drawable: drawable)
        }
    }

    private func render(_ texture: MTLTexture, generated: Bool, upscaled: Bool, size: CGSize,
                        drawable: CAMetalDrawable) {
        guard let command = resources.queue.makeCommandBuffer() else {
            available.signal(); return
        }
        lock.lock()
        let shouldPresent = active && self.size == size
            && drawable.texture.width == Int(size.width.rounded())
            && drawable.texture.height == Int(size.height.rounded())
        lock.unlock()
        guard shouldPresent else { available.signal(); return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
        guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
            available.signal(); return
        }
        encoder.setRenderPipelineState(resources.pipeline)
        let ratio = min(Double(drawable.texture.width) / Double(texture.width),
                        Double(drawable.texture.height) / Double(texture.height))
        let w = Double(texture.width) * ratio, h = Double(texture.height) * ratio
        encoder.setViewport(MTLViewport(originX: (Double(drawable.texture.width) - w) / 2,
                                        originY: (Double(drawable.texture.height) - h) / 2,
                                        width: w, height: h, znear: 0, zfar: 1))
        encoder.setFragmentTexture(texture, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        drawable.addPresentedHandler { [weak self] presented in
            guard let self, presented.presentedTime > 0 else { return }
            self.lock.lock(); let deliver = self.active; self.lock.unlock()
            let time = presented.presentedTime
            if deliver { DispatchQueue.main.async { self.onPresented(generated, time, upscaled) } }
        }
        command.addCompletedHandler { [weak self, available] completed in
            available.signal()
            if completed.status == .error, let self {
                let message = "Window presentation failed: \(completed.error?.localizedDescription ?? "Metal command error")"
                DispatchQueue.main.async { self.onError(message) }
            }
        }
        command.commit()
        // The display-link callback owns this deadline. Notify it on the same
        // render thread after committing work, without a scheduled-handler hop.
        drawable.present()
        DispatchQueue.main.async { [weak self] in self?.onSubmitted() }
    }
}

/// Carbon hotkeys do not require Accessibility/Input Monitoring permission. Unlike
/// Command-Option-Escape, this chord does not conflict with macOS Force Quit.
@MainActor
private final class CaptureStopHotkey {
    private var reference: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let action: () -> Void

    init(action: @escaping () -> Void) throws {
        self.action = action
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let pointer = Unmanaged.passUnretained(self).toOpaque()
        let install = InstallEventHandler(GetApplicationEventTarget(), { _, _, data in
            guard let data else { return noErr }
            let object = Unmanaged<CaptureStopHotkey>.fromOpaque(data).takeUnretainedValue()
            MainActor.assumeIsolated { object.action() }
            return noErr
        }, 1, &spec, pointer, &handler)
        guard install == noErr else { throw CaptureEngineError.message("Could not install the emergency stop shortcut (\(install)).") }
        let result = RegisterEventHotKey(UInt32(kVK_ANSI_N), UInt32(controlKey | optionKey | cmdKey),
            EventHotKeyID(signature: 0x4E45544C, id: 1), GetApplicationEventTarget(), 0, &reference)
        guard result == noErr else {
            if let handler { RemoveEventHandler(handler); self.handler = nil }
            throw CaptureEngineError.message("The stop shortcut ⌃⌥⌘N is unavailable (\(result)). Free that shortcut before starting Window mode.")
        }
    }

    deinit {
        if let reference { UnregisterEventHotKey(reference) }
        if let handler { RemoveEventHandler(handler) }
    }
}
