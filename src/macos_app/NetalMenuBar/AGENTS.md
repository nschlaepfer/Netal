# Netal menu-bar design

Build on the existing app. The user-approved visual baseline is
`docs/design-reference/approved-netal-menubar.png` at the repository root.

- Preserve the original compact 300-point panel, five-tab navigation, status header, native macOS surfaces, and existing settings arrangement. Add capabilities within that design; do not replace it with a new dashboard as part of polish.
- Netal's blob is part of the product. Keep it in the menu bar and the animated About panel, including the original glow, hover, hop, and landing behavior. Preserve the existing blob images and measured eye resources.
- The generated chrome `NetalChrome.png` is the application icon. It must not replace the blob with a wordmark, an N glyph, or another character inside the menu-bar experience.
- Keep the native glass appearance supplied by the macOS popover. Match existing controls and spacing for new features. Respect Reduce Motion and Reduce Transparency.
- Keep native model SR/FG and experimental window enhancement distinct. Discovery, settings, initialization, or synthetic tests do not establish live gameplay or universal compatibility.
- Prioritize Netal's own native Metal FG runtime/model. Apple VideoToolbox FG is an optional research comparator, not the product implementation target. Improvements to the capture-based block-motion path do not validate the native neural FG model.

See `docs/ORACULUM_DESIGN_LANGUAGE.md` and `docs/WINDOW_MODE.md` at the repository root.
