# Does the ANE take it?

Answers one question, the one that gates every ANE plan: **which operations will Core ML actually
place on the Neural Engine**. Not a stopwatch and not a power counter — the plan the runtime will
execute, read straight out of `MLComputePlan`. Public API, no private `_ANEClient`/`_ANECompiler`,
no sudo.

    python3 -m venv venv && ./venv/bin/pip install coremltools numpy   # needs Python <= 3.12
    ./venv/bin/python build-blocks.py                                  # writes sw-*.mlpackage
    xcrun swiftc -O -parse-as-library placement.swift -o placement
    ./placement sw-*.mlpackage

`build-blocks.py` emits one PWin-shaped transformer block per layer width in the real network, with
and without the shifted window. `placement.swift` prints, per model, how many operations land on the
ANE and — for the ones that do not — whether the ANE was merely not preferred or not capable at all.
That distinction is the whole point: "can run on: CPU/ANE" is a scheduling choice, "can run on: CPU"
is a wall.

Measured on M3 Max / macOS 27.0 — see `docs/performance.md`, "What the ANE will actually take".

Two traps worth inheriting. Build the models with **Python 3.12 or older**: on 3.14 coremltools
imports but fails at save with `BlobWriter not loaded`. And size the probe like the real layer — the
first version of this used 1024 tokens, got 0/15 on the ANE, and would have closed the question with
exactly the wrong answer.
