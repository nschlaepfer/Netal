# FG initial merge reconstruction unit

This directory reconstructs the first dependency-closed live-selected Frame
Generation unit: evaluate-1 launch L56, `k_initial_merge`, from module token 53.

The RTX program loads the exact 912-byte selected module through the CUDA
Driver API and dispatches its original ABI and geometry. The Mac program uses
a separately implemented deterministic input generator and a Metal kernel
that preserves the original five 32-bit stores at byte offsets 0, 4, 8, 12,
and 16 of each 20-byte record.

Neither executable accepts reference output or input snapshots as execution
inputs. Each generates A/B/C locally, poison-fills the output, executes, reads
all resources back, proves input immutability and canary integrity, and writes
an address-free run receipt. Cross-backend comparison happens only afterward.
