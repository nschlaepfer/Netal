// The exact accepted module13 arithmetic and three-pass signed-64 atomic
// lowering are shape-parameterized. This wrapper deliberately reuses that
// immutable implementation; dynamic extents and live ABI scalars are supplied
// only by the new host.
#include "../fg_early_l5_l15/early_l5_l15.metal"
