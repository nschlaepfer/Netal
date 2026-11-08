"""Validated opt-in descriptor conversion for the approximate MPP GEMM candidate.

No weights are changed. Non-GEMM operators retain their original implementations.
The 16-u32 ABI is shared by pwin_gemm_mpp.metal and the synthetic GPU test.
"""
from copy import deepcopy

LIBRARY = "pwin_mpp_all"


def validate_params(params):
    if len(params) != 16 or any(type(v) is not int or not 0 <= v <= 0xffffffff for v in params):
        raise ValueError("MPP parameters must be exactly 16 unsigned 32-bit integers")
    tokens, k, n, xs, ys, cs, epi, cmode, split, width, shift, ws, cols, rs, rf, reserved = params
    limit = cols or n
    if not (0 < tokens <= 0x3fffffff and 0 < k <= 640 and k % 16 == 0 and 0 < n <= 768 and n % 16 == 0):
        raise ValueError(f"unsupported MPP shape tokens={tokens}, K={k}, N={n}; require K<=640/N<=768 in multiples of 16")
    if width == 0 or xs < k or (ws or n) < n or not 0 < limit <= n or ys < limit:
        raise ValueError("MPP width, input/weight/output strides or store_cols are invalid")
    if epi not in range(4) or cmode not in (0, 1) or rf not in (0, 1) or reserved:
        raise ValueError("unsupported MPP epilogue, C mode, row filter or reserved word")
    mode, rounds = split & 15, split >> 4
    if mode > 6 or (rounds and (mode not in (5, 6) or rounds >= k // 16)):
        raise ValueError(f"unsupported MPP split schedule {split}")
    if (cs != 0 and cs < n) or (cmode == 1 and cs < n) or (epi == 3 and rs < limit):
        raise ValueError("MPP C/residual stride is smaller than the accessed columns")
    if (k - 1) * (ws or n) + n > 0xffffffff or (cmode == 0 and 15 * cs + n > 0xffffffff):
        raise ValueError("MPP weight/bias indexing exceeds uint32")
    for stride in (xs, ys, cs if cmode else 0, rs if epi == 3 else 0):
        if tokens * stride > 0xffffffff:
            raise ValueError("MPP tensor indexing exceeds uint32")
    return params


def dispatch(params, buffers):
    """Build a checked MPP dispatch; grid.y replaces reference SIMD N splitting."""
    validate_params(params)
    tokens, k, n = params[:3]
    return {"function": "pwin_gemm_mpp", "grid": [(tokens + 63) // 64, ((params[12] or n) + 31) // 32, 1],
            "block": [128, 1, 1], "buffers": buffers,
            "bytes": [{"index": 4, "u32": params}], "constants": {"0": k, "1": n}}


def convert_pipeline(pipeline):
    """Convert every GEMM3 dispatch, rejecting mixed/unknown GEMM variants.

    Fused MLP, attention, normalization and mixing remain the original kernels.
    They are present in the candidate library, not silently presented as MPP.
    """
    q = deepcopy(pipeline)
    count = 0
    for index, d in enumerate(q["dispatches"]):
        function = d["function"]
        if function == "pwin_gemm3_tokens":
            c = d["constants"]
            required = {"0", "1", "2", "3", "7", "8", "9"}
            if not required <= set(c) or set(c) - required - {"16"}:
                raise ValueError(f"dispatch {index}: unsupported GEMM3 function constants")
            nsplit = c.get("16", 1) or 1
            if type(nsplit) is not int or nsplit <= 0 or c["1"] % nsplit or (c["1"] // nsplit) % 16:
                raise ValueError(f"dispatch {index}: invalid reference N split")
            if {b["index"] for b in d["buffers"]} != {0,1,2,3,5,6}:
                raise ValueError(f"dispatch {index}: unsupported GEMM3 buffer ABI")
            words = [b["u32"] for b in d.get("bytes", []) if b["index"] == 4]
            if len(d["bytes"]) != 1 or len(words) != 1 or len(words[0]) != 8:
                raise ValueError(f"dispatch {index}: unsupported GEMM3 parameter ABI")
            t, epi, cmode, split, width, shift, cols, rf = words[0]
            params = [t, c["0"], c["1"], c["2"], c["7"], c["3"], epi, cmode,
                      split, width, shift, c["8"], cols, c["9"], rf, 0]
            try:
                converted = dispatch(params, d["buffers"])
            except ValueError as error:
                raise ValueError(f"dispatch {index}: {error}") from error
            # Preserve optional profiling metadata, but replace the entire ABI.
            d.update(converted)
            count += 1
        elif function.startswith("pwin_gemm"):
            raise ValueError(f"dispatch {index}: unsupported GEMM function {function}")
    if count == 0:
        raise ValueError("MPP profile contains no convertible GEMM3 dispatches")
    q["library"] = LIBRARY
    q["gemm_backend"] = "mpp"
    q["candidate_approximate"] = True
    q["reference_exact"] = False
    return q
