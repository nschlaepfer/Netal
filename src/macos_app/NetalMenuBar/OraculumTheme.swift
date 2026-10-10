import SwiftUI
import AppKit

/// Native presentation vocabulary shared by Oraculum research apps. Content remains quiet;
/// Liquid Glass is reserved for navigation and controls floating above that content.
enum OraculumTheme {
    static let panelWidth: CGFloat = 360
    static let inset: CGFloat = 14
    static let cornerRadius: CGFloat = 10
    static let accent = Color.accentColor
    static let border = Color.primary.opacity(0.06)
    static let surface = Color(nsColor: .controlBackgroundColor)
    // AppKit bitmap caching cannot capture the compositor's live glass surfaces.
    // Layout receipts deliberately use the real older-macOS material fallback.
    static var isBitmapPreview: Bool { CommandLine.arguments.contains("--render") }
}

/// A semantic section rather than another box. Preserve the compact native form layout.
struct OraculumCard<Content: View>: View {
    @ViewBuilder var content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        content.padding(.vertical, 4)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct OraculumGlassContainer<Content: View>: View {
    var spacing: CGFloat = 8
    @ViewBuilder var content: Content
    init(spacing: CGFloat = 8, @ViewBuilder content: () -> Content) {
        self.spacing = spacing
        self.content = content()
    }
    @ViewBuilder var body: some View {
        if #available(macOS 26.0, *), !OraculumTheme.isBitmapPreview {
            GlassEffectContainer(spacing: spacing) { content }
        } else {
            content
        }
    }
}

extension View {
    /// Actual system Liquid Glass on macOS 26; no gradient or translucent-color imitation.
    /// Native materials remain the fallback for supported older macOS releases.
    func oraculumGlass(cornerRadius: CGFloat = 12, tint: Color? = nil,
                       interactive: Bool = false) -> some View {
        modifier(OraculumGlassSurface(cornerRadius: cornerRadius, tint: tint, interactive: interactive))
    }

    func oraculumGlassButton(prominent: Bool = false) -> some View {
        modifier(OraculumGlassButton(prominent: prominent))
    }
}

private struct OraculumGlassSurface: ViewModifier {
    let cornerRadius: CGFloat
    let tint: Color?
    let interactive: Bool
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    @ViewBuilder func body(content: Content) -> some View {
        if #available(macOS 26.0, *), !reduceTransparency, !OraculumTheme.isBitmapPreview {
            content.glassEffect(.regular.tint(tint).interactive(interactive),
                                in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        } else if reduceTransparency {
            content.background(OraculumTheme.surface,
                               in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        } else {
            content.background(.regularMaterial,
                               in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        }
    }
}

private struct OraculumGlassButton: ViewModifier {
    let prominent: Bool
    @Environment(\.isEnabled) private var isEnabled

    @ViewBuilder func body(content: Content) -> some View {
        // Explicit semantic label colors remain readable when an offscreen snapshot cannot
        // reproduce the compositor's glass surface. Button behavior and accessibility stay native.
        let labeled = content.foregroundStyle(isEnabled ? Color.primary : Color.secondary)
        // Match the user's approved native popover controls. Its window provides the glass;
        // buttons retain the original bordered appearance instead of adding another glass layer.
        if prominent { labeled.buttonStyle(.borderedProminent) }
        else { labeled.buttonStyle(.bordered) }
    }
}
