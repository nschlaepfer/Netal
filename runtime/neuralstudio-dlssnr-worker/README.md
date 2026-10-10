# NeuralStudio DLSSNR worker shim

This private archival directory contains the exact source-built bridge binaries
used by the isolated-worker acceptance gate. They are deliberately separate
from `runtime/streamline-2.13`, which remains the unmodified NVIDIA-signed
runtime archive.

`sl.interposer.dll` and `sl.common.dll` were built from Streamline commit
`e8aaa6eaac968711fb62473d4ae8256dde20919b` with the three patches in
`patches/`. `neuralstudio/nvngx.dll` is built from
`src/media_host/ngx_population_thunk.cpp`. The exact hashes are in
`MANIFEST.sha256`.

Streamline retains NVIDIA's upstream MIT copyright and license. The
NeuralStudio patches and forwarding thunk are repository source under the
top-level MIT license. These development binaries are not Authenticode-signed
and must never replace or be represented as NVIDIA-signed files.

The staging script combines these three source-built files with the separately
archived, NVIDIA-signed `sl.dlss_nr.dll` and `nvngx_dlssnr.dll` in a build-only
directory. Nothing here is installed into a game or system directory.
