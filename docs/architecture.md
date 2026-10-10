# Architecture

Netal has three separable parts: the menu-bar app, the game bridge, and locally provisioned runtime bundles.

## App

`src/macos_app/NetalMenuBar` contains the SwiftUI views, log parser, file monitor, model integrity checks, and launch controls. Runtime paths are resolved from an environment override, user-selected runtime folder, or the checkout containing a source-built app. The app observes logs and writes control markers; it does not itself run the neural networks.

`RuntimeProcess` runs helper scripts from their checkout root, continuously drains output, and delivers the final exit status and a bounded output tail. Import/profile tools must report explicit success before the app labels the operation complete.

## SR and FG

The native bridge is arm64; the CrossOver bridge is x86_64 for the Wine host. Both use the SR texture adapter and session, and link the native FG runtime. The bridge observes and replaces supported MetalFX encode calls, with the original path retained for declined frames.

The SR bundle contains model data, Metal kernels, geometry and pipeline descriptions, and arithmetic tables. The FG bundle contains its native archive, Metal libraries, model tokens, and tables. The currently recorded FG profile is 640×360 internally and resampled to the game's output.

The CrossOver helper currently installs temporary hook files and compiles checkout-specific paths into the Wine helper. These are developer integration constraints; portable helper packaging is still a release gate.

## Experimental NR

Native NR source remains a research implementation. MLX-DLSS is a candidate external backend, not a bundled or selectable game runtime in this preview. Normal SR/FG launches do not enable NR. Numerical reference checks, temporal image quality, and watchdog-safe continuous presentation must be established independently.

## Public source boundary

The public exporter retains allowlisted source and app artwork. It excludes the Git history, builds, model data, captured evidence, compiled libraries, and machine-specific helpers. Source files that require excluded inputs remain research tools and are not promised to build from the public snapshot alone. Consult `SOURCE_MANIFEST.json` for exactly what was exported.
