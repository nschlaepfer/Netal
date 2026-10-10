// Build the macOS app icon from the transparent brand mark.
//
// The previous icon was `docs/brand/netal-icon.png` scaled straight into the iconset: a photographic
// render on an opaque white field, filling the square edge to edge. macOS draws app icons on the
// assumption that the artwork supplies its own rounded-rect body inside a larger canvas, so a full
// square reads as a white tile with something in it rather than as an icon.
//
// This draws the body macOS expects: an 824pt rounded rect centred in a 1024pt canvas (the Big Sur
// grid), a shadow beneath it, and the mark inset within that. The margin is not wasted space - it is
// where the shadow lives, and it is what makes the icon sit at the same visual size as every other
// icon in the Dock.
//
//   swift tools/netal-appicon.swift <mark.png> <out.png>

import AppKit
import CoreGraphics
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    FileHandle.standardError.write("usage: netal-appicon.swift <mark.png> <out.png>\n".data(using: .utf8)!)
    exit(64)
}
guard let mark = NSImage(contentsOfFile: arguments[1]),
      let markCG = mark.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("cannot read \(arguments[1])\n".data(using: .utf8)!)
    exit(1)
}

let canvas: CGFloat = 1024
// Big Sur's grid: the body is 824 of 1024, leaving 100 a side for the shadow to fall into.
let body: CGFloat = 824
let radius: CGFloat = body * 0.2237   // Apple's continuous-corner proportion, close enough as an arc

let space = CGColorSpaceCreateDeviceRGB()
guard let context = CGContext(data: nil, width: Int(canvas), height: Int(canvas), bitsPerComponent: 8,
                              bytesPerRow: 0, space: space,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    FileHandle.standardError.write("cannot create context\n".data(using: .utf8)!)
    exit(1)
}

let origin = (canvas - body) / 2
let bodyRect = CGRect(x: origin, y: origin, width: body, height: body)
let bodyPath = CGPath(roundedRect: bodyRect, cornerWidth: radius, cornerHeight: radius, transform: nil)

// Shadow under the body, slightly below centre, as macOS icons carry.
context.saveGState()
context.setShadow(offset: CGSize(width: 0, height: -18), blur: 44,
                  color: CGColor(red: 0, green: 0, blue: 0, alpha: 0.34))
context.addPath(bodyPath)
context.setFillColor(CGColor(red: 0.09, green: 0.10, blue: 0.13, alpha: 1))
context.fillPath()
context.restoreGState()

// The field: a deep slate that lets a pearl-grey mark and its warm eyes carry the icon. Dark rather
// than light because the mark is itself light, and a light-on-light icon disappears in the Dock.
context.saveGState()
context.addPath(bodyPath)
context.clip()
let gradient = CGGradient(colorsSpace: space, colors: [
    CGColor(red: 0.16, green: 0.18, blue: 0.23, alpha: 1),
    CGColor(red: 0.07, green: 0.08, blue: 0.11, alpha: 1),
] as CFArray, locations: [0, 1])!
context.drawLinearGradient(gradient, start: CGPoint(x: 0, y: canvas),
                           end: CGPoint(x: 0, y: origin), options: [])

// A soft highlight across the top, so the body reads as a surface rather than a flat swatch.
let sheen = CGGradient(colorsSpace: space, colors: [
    CGColor(red: 1, green: 1, blue: 1, alpha: 0.10),
    CGColor(red: 1, green: 1, blue: 1, alpha: 0.0),
] as CFArray, locations: [0, 1])!
context.drawLinearGradient(sheen, start: CGPoint(x: 0, y: origin + body),
                           end: CGPoint(x: 0, y: origin + body * 0.55), options: [])
context.restoreGState()

// The mark, inset within the body and sitting a little above centre - optical centring, because the
// shape is bottom-heavy and geometric centring makes it look like it is sliding off.
let markAspect = CGFloat(markCG.width) / CGFloat(markCG.height)
let markWidth = body * 0.68
let markHeight = markWidth / markAspect
let markRect = CGRect(x: (canvas - markWidth) / 2,
                      y: (canvas - markHeight) / 2 + body * 0.015,
                      width: markWidth, height: markHeight)
context.saveGState()
context.setShadow(offset: CGSize(width: 0, height: -10), blur: 30,
                  color: CGColor(red: 0, green: 0, blue: 0, alpha: 0.45))
context.draw(markCG, in: markRect)
context.restoreGState()

guard let out = context.makeImage() else { exit(1) }
let rep = NSBitmapImageRep(cgImage: out)
guard let data = rep.representation(using: .png, properties: [:]) else { exit(1) }
try! data.write(to: URL(fileURLWithPath: arguments[2]))
print("wrote \(arguments[2]) (\(Int(canvas))x\(Int(canvas)), body \(Int(body)), radius \(Int(radius)))")
