#!/usr/bin/env python3
"""Find a DLSS model inside an nvngx_dlss.dll.

The goal is that a user supplies a DLL and Netal takes it from there. This is the tool that decides
whether that is achievable, and it is not a guess: Netal already has a known-good model, so the
question is not "find something that looks like weights" but "do these exact bytes appear in that
file, and in what form".

That is a search with an answer key, which makes it cheap and unambiguous. Three outcomes:

  found, verbatim     the weights are stored plainly. Extraction is seek-and-read, and the rest of
                      the ingest pipeline is a day's work.
  found, encoded      stored, but transformed - byte order, or fp32 that the loader narrows. Still
                      straightforward: the transform is known once it matches.
  not found           either the DLL packs or compresses them, or this DLL is a different DLSS
                      version from the reference. The structural scan below distinguishes those:
                      if the file contains a large region whose value distribution looks like fp16
                      weights, the data is there in some form; if it contains nothing of the kind,
                      it is packed or absent.

    tools/netal-dlss-probe.py <nvngx_dlss.dll> [--reference <bundle>]

Nothing here needs to understand NVIDIA's loader, and nothing is redistributed: it reads a file the
user already has and reports offsets.
"""
import argparse, mmap, pathlib, struct, sys

REFERENCE_DEFAULT = pathlib.Path("work/re9-sr-runtime-bundle")
# A window long enough to be unique in a hundred-megabyte binary, taken from a busy part of the
# reference rather than its start, which may be zero-padded and would match everywhere.
WINDOW = 512


def encodings(raw: bytes) -> list[tuple[str, bytes]]:
    """The forms the same weights could plausibly take on disk."""
    out = [("verbatim fp16", raw)]

    swapped = bytearray(len(raw))
    swapped[0::2] = raw[1::2]
    swapped[1::2] = raw[0::2]
    out.append(("byte-swapped fp16", bytes(swapped)))

    # Stored as fp32 and narrowed at load time. Four bytes per value instead of two, so the needle
    # is twice the size; the search is identical.
    values = struct.unpack("<%de" % (len(raw) // 2), raw[: len(raw) // 2 * 2])
    out.append(("fp32", struct.pack("<%df" % len(values), *values)))
    return out


def busy_window(data: bytes, size: int) -> tuple[int, bytes]:
    """A slice with no long run of repeated bytes, so a hit means something."""
    step = max(1, len(data) // 64)
    for start in range(len(data) // 4, len(data) - size, step):
        chunk = data[start : start + size]
        if len(set(chunk)) > 64:
            return start, chunk
    middle = len(data) // 2
    return middle, data[middle : middle + size]


def search(haystack: mmap.mmap, name: str, needle: bytes) -> int | None:
    offset, window = busy_window(needle, WINDOW)
    position = haystack.find(window)
    while position != -1:
        start = position - offset
        if start >= 0 and start + len(needle) <= len(haystack):
            if haystack[start : start + len(needle)] == needle:
                return start
        position = haystack.find(window, position + 1)
    return None


def looks_like_weights(sample: bytes) -> tuple[bool, float, float]:
    """Do these bytes read as fp16 network weights?

    Trained weights are a narrow, roughly symmetric distribution about zero with no infinities.
    Code, strings and image data are none of those, so this separates a weight blob from the rest
    of a binary without knowing anything about the format around it.
    """
    count = len(sample) // 2
    if count < 256:
        return False, 0.0, 0.0
    values = struct.unpack("<%de" % count, sample[: count * 2])
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if len(finite) < count * 0.99:
        return False, 0.0, 0.0
    magnitude = sum(abs(v) for v in finite) / len(finite)
    positive = sum(1 for v in finite if v > 0) / max(1, len(finite))
    plausible = 0.0005 < magnitude < 1.5 and 0.35 < positive < 0.65
    return plausible, magnitude, positive


def scan(haystack: mmap.mmap, block: int = 65536) -> list[tuple[int, int]]:
    """Contiguous regions that read as fp16 weights, longest first."""
    runs, start = [], None
    for offset in range(0, len(haystack) - block, block):
        ok, _, _ = looks_like_weights(haystack[offset : offset + 4096])
        if ok and start is None:
            start = offset
        elif not ok and start is not None:
            runs.append((start, offset - start))
            start = None
    if start is not None:
        runs.append((start, len(haystack) - start))
    runs.sort(key=lambda r: -r[1])
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dll", type=pathlib.Path)
    parser.add_argument("--reference", type=pathlib.Path, default=REFERENCE_DEFAULT)
    args = parser.parse_args()

    if not args.dll.is_file():
        print("no such file: %s" % args.dll, file=sys.stderr)
        return 2

    model = args.reference / "model"
    references = {}
    for name in ("weights", "constants"):
        path = model / ("%s.bin" % name)
        if path.is_file():
            references[name] = path.read_bytes()
    if not references:
        print("no reference model under %s" % model, file=sys.stderr)
        return 2

    print("dll        %s (%.1f MB)" % (args.dll, args.dll.stat().st_size / 1048576))
    for name, raw in references.items():
        print("reference  %s.bin %d bytes" % (name, len(raw)))

    with open(args.dll, "rb") as handle:
        data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)

        hit = False
        for name, raw in references.items():
            for label, encoded in encodings(raw):
                found = search(data, name, encoded)
                if found is not None:
                    hit = True
                    print("FOUND      %s.bin as %s at offset 0x%x (%d bytes)"
                          % (name, label, found, len(encoded)))
                    break
            else:
                print("absent     %s.bin in any tried encoding" % name)

        if hit:
            print("result=extractable")
            return 0

        # Nothing matched. Either this is a different DLSS version from the reference - in which
        # case the weights differ but are still stored the same way - or they are packed. A
        # structural scan tells those apart.
        print("\nno exact match; scanning for regions shaped like fp16 weights")
        runs = scan(data)
        if not runs:
            print("  none — the model is packed, compressed, or not in this file")
            print("result=not-plain")
            return 1
        for offset, length in runs[:5]:
            ok, magnitude, positive = looks_like_weights(data[offset : offset + 4096])
            print("  0x%08x  %8.2f MB  mean |w| %.4f  %.0f%% positive"
                  % (offset, length / 1048576, magnitude, positive * 100))
        print("result=candidates  (a different DLSS version stores weights here; extract and "
              "prove them against the RTX receipts)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
