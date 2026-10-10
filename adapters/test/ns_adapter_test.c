/*
 * NeuralStudio test reference adapter
 * SPDX-License-Identifier: MIT
 *
 * Implements the v1 adapter ABI with a deterministic CPU transform. Its purpose
 * is not image quality: it is to let the entire host pipeline - source, guide
 * layer, broker, adapter hosting, output, telemetry, and failure isolation - run
 * end to end in CI with no proprietary runtime and no GPU.
 *
 * Deliberate properties:
 *
 *   - Deterministic. The same inputs and settings always produce the same
 *     output, so conformance tests can assert on checksums.
 *   - Stateless. Exercises the broker's stateless path; the NGX adapter covers
 *     the stateful path.
 *   - Failure-injectable. SimulateFailureAfterFrames makes the host's degrade,
 *     bypass, and isolation paths testable on demand rather than by accident.
 *
 * Scope limit for v1: CPU resources only. probe() reports d3d12 as unsupported
 * with an explicit reason rather than pretending, because a capability report
 * that says "unsupported" without a reason is the failure mode this project has
 * already paid for once.
 */

#define NS_ADAPTER_IMPLEMENTATION 1
#include "../../spec/adapter/v1/neuralstudio_adapter.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define NS_TEST_PACKAGE_ID "org.neuralstudio.test.passthrough"
#define NS_TEST_VERSION "1.0.0"

/* ------------------------------------------------------------- half float - */

static float ns_half_to_float(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1Fu;
    const uint32_t mantissa = h & 0x3FFu;
    uint32_t bits;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            /* Subnormal: normalize it. */
            uint32_t e = exponent;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                e--;
            }
            m &= 0x3FFu;
            bits = sign | ((e + 127u - 15u + 1u) << 23) | (m << 13);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t ns_float_to_half(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        return (uint16_t)sign;
    }
    if (exponent >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

/* ---------------------------------------------------------------- session - */

typedef enum ns_test_mode {
    ns_test_mode_identity = 0,
    ns_test_mode_invert = 1,
    ns_test_mode_luminance = 2,
    ns_test_mode_frame_tint = 3
} ns_test_mode;

struct ns_session_t {
    ns_extent2d extent;
    ns_format color_format;
    ns_format output_format;
    float strength;
    int32_t mode;
    int32_t simulate_failure_after_frames;
    uint64_t evaluations;
};

static const ns_host_v1 *g_host = NULL;
static int g_initialized = 0;

static void ns_log(ns_log_level level, const char *message)
{
    if (g_host != NULL && g_host->log != NULL) {
        g_host->log(g_host->user_data, level, message);
    }
}

static void ns_copy_detail(char *destination, size_t capacity, const char *text)
{
    if (destination == NULL || capacity == 0u) {
        return;
    }
    if (text == NULL) {
        destination[0] = '\0';
        return;
    }
#if defined(_MSC_VER)
    strncpy_s(destination, capacity, text, _TRUNCATE);
#else
    strncpy(destination, text, capacity - 1u);
    destination[capacity - 1u] = '\0';
#endif
}

static int ns_format_supported(ns_format format)
{
    return format == ns_format_r8g8b8a8_unorm
        || format == ns_format_b8g8r8a8_unorm
        || format == ns_format_r16g16b16a16_float;
}

static size_t ns_bytes_per_pixel(ns_format format)
{
    switch (format) {
    case ns_format_r8g8b8a8_unorm:
    case ns_format_b8g8r8a8_unorm:
        return 4u;
    case ns_format_r16g16b16a16_float:
        return 8u;
    default:
        return 0u;
    }
}

/* --------------------------------------------------------------- exports - */

static ns_status NS_CALL test_get_info(ns_adapter_info *out_info)
{
    if (out_info == NULL || out_info->struct_size < sizeof(ns_adapter_info)) {
        return ns_status_invalid_argument;
    }
    out_info->package_id = NS_TEST_PACKAGE_ID;
    out_info->display_name = "NeuralStudio Test Reference";
    out_info->adapter_version = NS_TEST_VERSION;
    out_info->backend_name = "test";
    out_info->abi_version_major = NS_ADAPTER_ABI_VERSION_MAJOR;
    out_info->abi_version_minor = NS_ADAPTER_ABI_VERSION_MINOR;
    out_info->supports_sandboxed_hosting = 1u;
    out_info->requires_state_guard = 0u;
    return ns_status_ok;
}

static ns_status NS_CALL test_probe(const ns_device_desc *device, ns_capability_report *out_report)
{
    if (device == NULL || out_report == NULL || out_report->struct_size < sizeof(ns_capability_report)) {
        return ns_status_invalid_argument;
    }

    out_report->requires_signature_check = 0u;
    out_report->estimated_vram_bytes = 0u;

    if (device->api != ns_api_cpu) {
        out_report->supported = 0u;
        ns_copy_detail(out_report->detail, sizeof(out_report->detail),
                       "The test reference adapter implements CPU resources only in v1. "
                       "Open the source with api=cpu, or use a GPU-backed package for d3d12 sources.");
        return ns_status_ok;
    }

    out_report->supported = 1u;
    ns_copy_detail(out_report->detail, sizeof(out_report->detail),
                   "CPU reference backend available.");
    return ns_status_ok;
}

static ns_status NS_CALL test_initialize(const ns_host_v1 *host, const ns_init_desc *desc)
{
    if (host == NULL || desc == NULL) {
        return ns_status_invalid_argument;
    }
    if (desc->device.api != ns_api_cpu) {
        return ns_status_unsupported;
    }
    g_host = host;
    g_initialized = 1;
    ns_log(ns_log_info, "test reference adapter initialized");
    return ns_status_ok;
}

static void NS_CALL test_shutdown(void)
{
    g_initialized = 0;
    g_host = NULL;
}

static ns_status NS_CALL test_create_session(const ns_session_desc *desc, ns_session_handle *out_session)
{
    if (!g_initialized) {
        return ns_status_not_initialized;
    }
    if (desc == NULL || out_session == NULL || desc->struct_size < sizeof(ns_session_desc)) {
        return ns_status_invalid_argument;
    }
    if (!ns_format_supported(desc->color_format) || !ns_format_supported(desc->output_format)) {
        return ns_status_unsupported;
    }
    if (desc->input_extent.width != desc->output_extent.width ||
        desc->input_extent.height != desc->output_extent.height) {
        /* Declared as same-size in the manifest; refuse rather than silently rescale. */
        return ns_status_unsupported;
    }

    struct ns_session_t *session = (struct ns_session_t *)calloc(1u, sizeof(struct ns_session_t));
    if (session == NULL) {
        return ns_status_out_of_memory;
    }

    session->extent = desc->input_extent;
    session->color_format = desc->color_format;
    session->output_format = desc->output_format;
    session->strength = 1.0f;
    session->mode = ns_test_mode_identity;
    session->simulate_failure_after_frames = 0;
    session->evaluations = 0u;

    *out_session = session;
    return ns_status_ok;
}

static ns_status NS_CALL test_destroy_session(ns_session_handle session)
{
    if (session == NULL) {
        return ns_status_invalid_argument;
    }
    free(session);
    return ns_status_ok;
}

static ns_status NS_CALL test_reset_history(ns_session_handle session, ns_reset_reason reason)
{
    (void)reason;
    if (session == NULL) {
        return ns_status_invalid_argument;
    }
    /* Stateless: a reset is a no-op, but it must still succeed so the broker's
       reset path is exercised identically for stateless and stateful packages. */
    return ns_status_ok;
}

static void ns_transform_pixel(float *rgba, int32_t mode, float strength, uint64_t frame_index)
{
    float out[4];
    out[0] = rgba[0];
    out[1] = rgba[1];
    out[2] = rgba[2];
    out[3] = rgba[3];

    switch (mode) {
    case ns_test_mode_invert:
        out[0] = 1.0f - rgba[0];
        out[1] = 1.0f - rgba[1];
        out[2] = 1.0f - rgba[2];
        break;
    case ns_test_mode_luminance: {
        const float luma = 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2];
        out[0] = luma;
        out[1] = luma;
        out[2] = luma;
        break;
    }
    case ns_test_mode_frame_tint: {
        /* Varies with frame index so a stale-output bug is visible rather than
           silently passing a still-image comparison. */
        const float phase = (float)(frame_index % 60u) / 60.0f;
        out[0] = rgba[0] * (0.5f + 0.5f * phase);
        out[2] = rgba[2] * (1.0f - 0.5f * phase);
        break;
    }
    case ns_test_mode_identity:
    default:
        break;
    }

    rgba[0] = rgba[0] + (out[0] - rgba[0]) * strength;
    rgba[1] = rgba[1] + (out[1] - rgba[1]) * strength;
    rgba[2] = rgba[2] + (out[2] - rgba[2]) * strength;
    rgba[3] = out[3];
}

static ns_status NS_CALL test_evaluate(ns_session_handle session,
                                       const ns_frame *frame,
                                       ns_evaluate_result *out_result)
{
    if (session == NULL || frame == NULL || out_result == NULL) {
        return ns_status_invalid_argument;
    }
    if (out_result->struct_size < sizeof(ns_evaluate_result)) {
        return ns_status_invalid_argument;
    }

    out_result->evaluated = 0u;
    out_result->output_fresh = 0u;
    out_result->used_fallback_path = 0u;
    out_result->inference_ms = 0.0;
    out_result->requested_reset = ns_reset_none;
    memset(&out_result->submission, 0, sizeof(out_result->submission));
    out_result->submission.struct_size = sizeof(ns_submission);
    out_result->detail[0] = '\0';

    if (session->simulate_failure_after_frames > 0 &&
        session->evaluations >= (uint64_t)session->simulate_failure_after_frames) {
        ns_copy_detail(out_result->detail, sizeof(out_result->detail),
                       "Simulated failure requested by SimulateFailureAfterFrames.");
        return ns_status_internal_error;
    }

    const ns_resource *color = NULL;
    for (uint32_t i = 0; i < frame->input_count; ++i) {
        if (frame->inputs[i].kind == ns_guide_color) {
            color = &frame->inputs[i];
            break;
        }
    }
    if (color == NULL) {
        ns_copy_detail(out_result->detail, sizeof(out_result->detail),
                       "No color input was supplied.");
        return ns_status_guide_missing;
    }
    if (color->api != ns_api_cpu || color->cpu_data == NULL ||
        frame->output.api != ns_api_cpu || frame->output.cpu_data == NULL) {
        ns_copy_detail(out_result->detail, sizeof(out_result->detail),
                       "This adapter requires CPU-accessible color input and output.");
        return ns_status_unsupported;
    }

    const size_t source_bpp = ns_bytes_per_pixel(color->format);
    const size_t destination_bpp = ns_bytes_per_pixel(frame->output.format);
    if (source_bpp == 0u || destination_bpp == 0u) {
        return ns_status_unsupported;
    }

    const uint32_t width = session->extent.width;
    const uint32_t height = session->extent.height;

    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t *source_row = (const uint8_t *)color->cpu_data + (size_t)y * color->row_pitch;
        uint8_t *destination_row = (uint8_t *)frame->output.cpu_data + (size_t)y * frame->output.row_pitch;

        for (uint32_t x = 0; x < width; ++x) {
            float rgba[4];

            if (color->format == ns_format_r16g16b16a16_float) {
                const uint16_t *source = (const uint16_t *)(source_row + (size_t)x * source_bpp);
                rgba[0] = ns_half_to_float(source[0]);
                rgba[1] = ns_half_to_float(source[1]);
                rgba[2] = ns_half_to_float(source[2]);
                rgba[3] = ns_half_to_float(source[3]);
            } else {
                const uint8_t *source = source_row + (size_t)x * source_bpp;
                const int b_first = (color->format == ns_format_b8g8r8a8_unorm);
                rgba[0] = (float)source[b_first ? 2 : 0] / 255.0f;
                rgba[1] = (float)source[1] / 255.0f;
                rgba[2] = (float)source[b_first ? 0 : 2] / 255.0f;
                rgba[3] = (float)source[3] / 255.0f;
            }

            ns_transform_pixel(rgba, session->mode, session->strength, frame->frame_index);

            if (frame->output.format == ns_format_r16g16b16a16_float) {
                uint16_t *destination = (uint16_t *)(destination_row + (size_t)x * destination_bpp);
                destination[0] = ns_float_to_half(rgba[0]);
                destination[1] = ns_float_to_half(rgba[1]);
                destination[2] = ns_float_to_half(rgba[2]);
                destination[3] = ns_float_to_half(rgba[3]);
            } else {
                uint8_t *destination = destination_row + (size_t)x * destination_bpp;
                const int b_first = (frame->output.format == ns_format_b8g8r8a8_unorm);
                for (int c = 0; c < 4; ++c) {
                    float v = rgba[c];
                    if (v < 0.0f) { v = 0.0f; }
                    if (v > 1.0f) { v = 1.0f; }
                    rgba[c] = v;
                }
                destination[b_first ? 2 : 0] = (uint8_t)(rgba[0] * 255.0f + 0.5f);
                destination[1] = (uint8_t)(rgba[1] * 255.0f + 0.5f);
                destination[b_first ? 0 : 2] = (uint8_t)(rgba[2] * 255.0f + 0.5f);
                destination[3] = (uint8_t)(rgba[3] * 255.0f + 0.5f);
            }
        }
    }

    session->evaluations++;
    out_result->evaluated = 1u;
    out_result->output_fresh = 1u;

    if (g_host != NULL && g_host->report_metric != NULL) {
        g_host->report_metric(g_host->user_data, "test.evaluations", (double)session->evaluations);
    }
    return ns_status_ok;
}

static ns_status NS_CALL test_set_setting(ns_session_handle session, const char *key, const ns_value *value)
{
    if (session == NULL || key == NULL || value == NULL) {
        return ns_status_invalid_argument;
    }

    if (strcmp(key, "Strength") == 0) {
        float v = (float)value->float_value;
        if (v < 0.0f || v > 1.0f) { return ns_status_invalid_argument; }
        session->strength = v;
        return ns_status_ok;
    }
    if (strcmp(key, "Mode") == 0) {
        if (value->int_value < 0 || value->int_value > 3) { return ns_status_invalid_argument; }
        session->mode = (int32_t)value->int_value;
        return ns_status_ok;
    }
    if (strcmp(key, "SimulateFailureAfterFrames") == 0) {
        if (value->int_value < 0) { return ns_status_invalid_argument; }
        session->simulate_failure_after_frames = (int32_t)value->int_value;
        return ns_status_ok;
    }
    return ns_status_unsupported;
}

static ns_status NS_CALL test_get_setting(ns_session_handle session, const char *key, ns_value *out_value)
{
    if (session == NULL || key == NULL || out_value == NULL) {
        return ns_status_invalid_argument;
    }

    if (strcmp(key, "Strength") == 0) {
        out_value->type = ns_value_float;
        out_value->float_value = (double)session->strength;
        return ns_status_ok;
    }
    if (strcmp(key, "Mode") == 0) {
        out_value->type = ns_value_int;
        out_value->int_value = session->mode;
        return ns_status_ok;
    }
    if (strcmp(key, "SimulateFailureAfterFrames") == 0) {
        out_value->type = ns_value_int;
        out_value->int_value = session->simulate_failure_after_frames;
        return ns_status_ok;
    }
    return ns_status_unsupported;
}

static ns_status NS_CALL test_query_vram(ns_session_handle session, uint64_t *out_bytes)
{
    if (session == NULL || out_bytes == NULL) {
        return ns_status_invalid_argument;
    }
    *out_bytes = 0u;
    return ns_status_ok;
}

static const char *NS_CALL test_status_string(ns_status status)
{
    switch (status) {
    case ns_status_ok: return "ok";
    case ns_status_unsupported: return "unsupported";
    case ns_status_invalid_argument: return "invalid argument";
    case ns_status_out_of_memory: return "out of memory";
    case ns_status_device_lost: return "device lost";
    case ns_status_runtime_missing: return "runtime missing";
    case ns_status_runtime_rejected: return "runtime rejected";
    case ns_status_not_initialized: return "not initialized";
    case ns_status_needs_reset: return "needs reset";
    case ns_status_needs_recreate: return "needs recreate";
    case ns_status_guide_missing: return "guide missing";
    case ns_status_not_implemented: return "not implemented";
    case ns_status_internal_error: return "internal error";
    default: return "unknown";
    }
}

static const ns_adapter_v1 g_adapter = {
    sizeof(ns_adapter_v1),
    NS_ADAPTER_ABI_VERSION_MAJOR,
    NS_ADAPTER_ABI_VERSION_MINOR,
    test_get_info,
    test_probe,
    test_initialize,
    test_shutdown,
    test_create_session,
    test_destroy_session,
    test_reset_history,
    test_evaluate,
    test_set_setting,
    test_get_setting,
    test_query_vram,
    test_status_string
};

NS_EXPORT const ns_adapter_v1 *NS_CALL nsGetAdapterV1(uint32_t requested_major)
{
    if (requested_major != NS_ADAPTER_ABI_VERSION_MAJOR) {
        return NULL;
    }
    return &g_adapter;
}
