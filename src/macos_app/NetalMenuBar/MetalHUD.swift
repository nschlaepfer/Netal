import Foundation

/// Apple's Metal performance HUD, toggled the way Apple documents it: the `MetalForceHudEnabled`
/// preference in the global domain, which is what `defaults write -g MetalForceHudEnabled -bool YES`
/// writes. Verified against this machine, where the HUD was already on and the key read back as 1.
///
/// CFPreferences rather than UserDefaults because the value belongs to `kCFPreferencesAnyApplication`
/// - every app - and `UserDefaults.standard` writes only into this app's own domain.
///
/// The HUD is created when a process starts Metal, so a change lands on the next launch rather than
/// on the running game. The UI says so instead of appearing to do nothing.
enum MetalHUD {
    private static let key = "MetalForceHudEnabled" as CFString

    static var isEnabled: Bool {
        let value = CFPreferencesCopyValue(key, kCFPreferencesAnyApplication,
                                           kCFPreferencesCurrentUser, kCFPreferencesAnyHost)
        if let number = value as? NSNumber { return number.boolValue }
        if let string = value as? String { return string == "1" || string.lowercased() == "yes" }
        return false
    }

    /// Returns false when the write did not stick - a sandbox or a managed preference can refuse it,
    /// and silently leaving the switch in the new position would be a lie.
    @discardableResult
    static func setEnabled(_ enabled: Bool) -> Bool {
        CFPreferencesSetValue(key, enabled as CFNumber, kCFPreferencesAnyApplication,
                              kCFPreferencesCurrentUser, kCFPreferencesAnyHost)
        CFPreferencesAppSynchronize(kCFPreferencesAnyApplication)
        let stuck = isEnabled == enabled
        if stuck { UserDefaults.standard.set(enabled, forKey: ownershipKey) }
        return stuck
    }

    // ---------------------------------------------------------------- borrowed, not taken

    /// Whether *Netal* turned the HUD on. Kept in Netal's own domain, deliberately: the HUD switch
    /// lives in everyone's domain, so without this there is no way to tell a HUD the user turned on
    /// themselves from one this app turned on.
    private static let ownershipKey = "NetalEnabledMetalHUD"

    /// Put the HUD back the way it was found.
    ///
    /// A switch in a menu bar app reads as belonging to that app, but this one writes a preference
    /// that outlives it - it survives quitting Netal, and it survives rebooting the Mac, in every
    /// Metal app on the machine. Flipping it once and finding the overlay in every game days later
    /// is not what anyone meant by that switch, so Netal gives it back when it leaves.
    ///
    /// Only when Netal was the one that set it. A HUD the user turned on themselves is theirs.
    static func relinquish() {
        guard UserDefaults.standard.bool(forKey: ownershipKey) else { return }
        CFPreferencesSetValue(key, false as CFNumber, kCFPreferencesAnyApplication,
                              kCFPreferencesCurrentUser, kCFPreferencesAnyHost)
        CFPreferencesAppSynchronize(kCFPreferencesAnyApplication)
        UserDefaults.standard.removeObject(forKey: ownershipKey)
    }
}
