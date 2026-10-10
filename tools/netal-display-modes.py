#!/usr/bin/env python3
"""Print every pixel resolution the main display can be set to, one WxH per line.

Netal keys its batched pipeline sets by output resolution, and a set that does not exist means the
frame silently runs the transpiled kernels - correct, but about twice as slow. A game's resolution
menu is just the display's mode list, so enumerating the modes tells us every output the user can
pick before they pick it.

CoreGraphics is called through ctypes rather than PyObjC, which is not present in the system
python3. Pixel width/height are used, not the point size: a Retina mode reports 1728x1080 points
for a 3456x2160 pixel target, and the pixel size is what MetalFX scales to.
"""
import ctypes, ctypes.util, sys

cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))
cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))

cg.CGMainDisplayID.restype = ctypes.c_uint32
cg.CGDisplayCopyAllDisplayModes.restype = ctypes.c_void_p
cg.CGDisplayCopyAllDisplayModes.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
cg.CGDisplayModeGetPixelWidth.restype = ctypes.c_size_t
cg.CGDisplayModeGetPixelWidth.argtypes = [ctypes.c_void_p]
cg.CGDisplayModeGetPixelHeight.restype = ctypes.c_size_t
cg.CGDisplayModeGetPixelHeight.argtypes = [ctypes.c_void_p]
cf.CFArrayGetCount.restype = ctypes.c_long
cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
cf.CFRelease.argtypes = [ctypes.c_void_p]

display = int(sys.argv[1]) if len(sys.argv) > 1 else cg.CGMainDisplayID()
modes = cg.CGDisplayCopyAllDisplayModes(display, None)
if not modes:
    sys.exit("no display modes reported for display %d" % display)

sizes = set()
for i in range(cf.CFArrayGetCount(modes)):
    mode = cf.CFArrayGetValueAtIndex(modes, i)
    sizes.add((cg.CGDisplayModeGetPixelWidth(mode), cg.CGDisplayModeGetPixelHeight(mode)))
cf.CFRelease(modes)

# A display mode is not the only output a game can ask for. Every Apple laptop display is taller
# than 16:9, and a game set to fullscreen 16:9 renders into a letterboxed 16:9 region of it: on this
# 3456x2234 screen, Resident Evil 4 scales to 3456x1944, which is not a mode and so was never
# generated. It then ran the transpiled kernels and spent 91 seconds compiling them, with nothing
# saying why. Emit the width-matched 16:9 companion of every mode alongside the mode itself.
#
# Only 16:9, and only where it differs from a real mode: it is the aspect every fullscreen game
# defaults to, each set costs about two seconds and 256K once, and guessing further would spend that
# on shapes nothing asks for.
for w, h in sorted(sizes | {(w, (w * 9 // 16) & ~1) for w, h in sizes if (w * 9 // 16) < h},
                   key=lambda s: (s[0] * s[1], s)):
    print("%dx%d" % (w, h))
