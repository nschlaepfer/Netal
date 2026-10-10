/*
 * NeuralStudio adapter conformance runner
 * SPDX-License-Identifier: MIT
 *
 * Loads an adapter through the documented export and asserts the behavior the
 * host relies on. Any adapter that ships must pass this, because the host's
 * safety and state-reporting rules are only as good as the adapter contract
 * underneath them.
 *
 * The checks that matter most:
 *
 *   - probe() must explain a negative result, not just return unsupported.
 *   - A missing required guide must be reported as guide_missing, never
 *     silently substituted with a neutral value.
 *   - evaluate() must set evaluated and output_fresh only when it really wrote
 *     output, because the product's Active state is derived from those flags.
 *   - Injected failures must surface as errors so the host can degrade.
 *
 * Usage: ns_conformance.exe <path-to-adapter.dll> [expected-package-id]
 *                           [--metadata-only]
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../spec/adapter/v1/neuralstudio_adapter.h"

static int g_checks = 0;
static int g_failures = 0;

static void check(int condition, const char *name, const char *detail)
{
    g_checks++;
    if (condition) {
        printf("  PASS  %s\n", name);
    } else {
        g_failures++;
        printf("  FAIL  %s%s%s\n", name, detail ? " - " : "", detail ? detail : "");
    }
}

static void host_log(void *user_data, ns_log_level level, const char *message)
{
    (void)user_data;
    printf("        [adapter %d] %s\n", (int)level, message ? message : "");
}

static void host_report_metric(void *user_data, const char *key, double value)
{
    (void)user_data; (void)key; (void)value;
}

static ns_status host_request_reset(void *user_data, ns_reset_reason reason)
{
    (void)user_data; (void)reason;
    return ns_status_ok;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: ns_conformance.exe <adapter.dll> [expected-package-id] [--metadata-only]\n");
        return 2;
    }

    const char *adapter_path = argv[1];
    const char *expected_package_id = (argc > 2) ? argv[2] : NULL;
    const int metadata_only = argc > 3 && strcmp(argv[3], "--metadata-only") == 0;

    printf("NeuralStudio adapter conformance\n");
    printf("adapter: %s\n\n", adapter_path);

    HMODULE module = LoadLibraryA(adapter_path);
    if (module == NULL) {
        fprintf(stderr, "LoadLibrary failed with %lu\n", GetLastError());
        return 2;
    }

    ns_get_adapter_v1_fn get_adapter =
        (ns_get_adapter_v1_fn)(void *)GetProcAddress(module, "nsGetAdapterV1");
    if (get_adapter == NULL) {
        fprintf(stderr, "adapter does not export nsGetAdapterV1\n");
        FreeLibrary(module);
        return 2;
    }

    printf("[export and version negotiation]\n");
    check(get_adapter(9999u) == NULL, "rejects an unknown ABI major version", NULL);

    const ns_adapter_v1 *adapter = get_adapter(NS_ADAPTER_ABI_VERSION_MAJOR);
    check(adapter != NULL, "returns a vtable for the supported ABI major version", NULL);
    if (adapter == NULL) {
        FreeLibrary(module);
        return 1;
    }
    check(adapter->struct_size >= sizeof(ns_adapter_v1), "vtable struct_size is at least the compiled size", NULL);
    check(adapter->abi_version_major == NS_ADAPTER_ABI_VERSION_MAJOR, "declares the expected ABI major", NULL);

    printf("\n[get_info]\n");
    ns_adapter_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    check(adapter->get_info(&info) == ns_status_ok, "get_info succeeds before initialize", NULL);
    check(info.package_id != NULL && info.package_id[0] != '\0', "reports a package id", NULL);
    if (expected_package_id != NULL && info.package_id != NULL) {
        check(strcmp(info.package_id, expected_package_id) == 0,
              "package id matches the manifest", info.package_id);
    }

    printf("\n[probe]\n");
    ns_device_desc cpu_device;
    memset(&cpu_device, 0, sizeof(cpu_device));
    cpu_device.struct_size = sizeof(cpu_device);
    cpu_device.api = ns_api_cpu;

    ns_capability_report report;
    memset(&report, 0, sizeof(report));
    report.struct_size = sizeof(report);
    check(adapter->probe(&cpu_device, &report) == ns_status_ok, "probe succeeds for a CPU device", NULL);
    if (metadata_only) {
        check(report.supported == 0u, "rejects an unsupported CPU device", report.detail);
        check(report.detail[0] != '\0', "explains why CPU is unsupported", NULL);
    } else {
        check(report.supported == 1u, "reports CPU support", report.detail);
    }

    ns_device_desc gpu_device;
    memset(&gpu_device, 0, sizeof(gpu_device));
    gpu_device.struct_size = sizeof(gpu_device);
    gpu_device.api = ns_api_d3d12;

    ns_capability_report gpu_report;
    memset(&gpu_report, 0, sizeof(gpu_report));
    gpu_report.struct_size = sizeof(gpu_report);
    adapter->probe(&gpu_device, &gpu_report);
    if (gpu_report.supported == 0u) {
        check(gpu_report.detail[0] != '\0',
              "explains why an unsupported device is unsupported", NULL);
    } else {
        check(1, "reports d3d12 support", NULL);
    }

    if (metadata_only) {
        printf("\n[metadata-only adapter surface]\n");
        check(info.supports_sandboxed_hosting == 1u,
              "declares sandboxed hosting support", NULL);
        check(adapter->initialize != NULL && adapter->create_session != NULL &&
                  adapter->evaluate != NULL && adapter->destroy_session != NULL,
              "exports the complete session lifecycle", NULL);
        check(adapter->status_string != NULL &&
                  strcmp(adapter->status_string(ns_status_device_lost),
                         "device_lost") == 0,
              "provides stable status text", NULL);
        FreeLibrary(module);
        printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
        return (g_failures == 0) ? 0 : 1;
    }

    printf("\n[initialize]\n");
    ns_host_v1 host;
    memset(&host, 0, sizeof(host));
    host.struct_size = sizeof(host);
    host.log = host_log;
    host.report_metric = host_report_metric;
    host.request_reset = host_request_reset;

    ns_init_desc init;
    memset(&init, 0, sizeof(init));
    init.struct_size = sizeof(init);
    init.device = cpu_device;
    init.package_root = ".";
    init.cache_root = ".";
    init.sandboxed = 0u;

    check(adapter->initialize(&host, &init) == ns_status_ok, "initialize succeeds", NULL);

    printf("\n[session lifecycle]\n");
    const uint32_t width = 64;
    const uint32_t height = 32;

    ns_session_desc session_desc;
    memset(&session_desc, 0, sizeof(session_desc));
    session_desc.struct_size = sizeof(session_desc);
    session_desc.input_extent.width = width;
    session_desc.input_extent.height = height;
    session_desc.output_extent.width = width;
    session_desc.output_extent.height = height;
    session_desc.color_format = ns_format_r8g8b8a8_unorm;
    session_desc.output_format = ns_format_r8g8b8a8_unorm;
    session_desc.conventions.struct_size = sizeof(ns_conventions);
    session_desc.package_root = ".";

    ns_session_handle session = NULL;
    check(adapter->create_session(&session_desc, &session) == ns_status_ok && session != NULL,
          "create_session succeeds", NULL);

    ns_session_desc mismatched = session_desc;
    mismatched.output_extent.width = width * 2u;
    ns_session_handle bad_session = NULL;
    check(adapter->create_session(&mismatched, &bad_session) != ns_status_ok,
          "refuses a resolution change the manifest does not declare", NULL);

    if (session == NULL) {
        adapter->shutdown();
        FreeLibrary(module);
        printf("\n%d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }

    check(adapter->reset_history(session, ns_reset_session_start) == ns_status_ok,
          "reset_history succeeds even for a stateless adapter", NULL);

    printf("\n[evaluate]\n");
    const size_t row_pitch = (size_t)width * 4u;
    const size_t buffer_bytes = row_pitch * height;
    uint8_t *input_pixels = (uint8_t *)malloc(buffer_bytes);
    uint8_t *output_pixels = (uint8_t *)malloc(buffer_bytes);
    if (input_pixels == NULL || output_pixels == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(input_pixels);
        free(output_pixels);
        adapter->destroy_session(session);
        adapter->shutdown();
        FreeLibrary(module);
        return 2;
    }
    for (size_t i = 0; i < buffer_bytes; ++i) {
        input_pixels[i] = (uint8_t)(i % 251u);
    }
    memset(output_pixels, 0, buffer_bytes);

    ns_resource color;
    memset(&color, 0, sizeof(color));
    color.struct_size = sizeof(color);
    color.kind = ns_guide_color;
    color.format = ns_format_r8g8b8a8_unorm;
    color.api = ns_api_cpu;
    color.extent.width = width;
    color.extent.height = height;
    color.cpu_data = input_pixels;
    color.row_pitch = row_pitch;

    ns_resource output = color;
    output.cpu_data = output_pixels;

    ns_frame frame;
    memset(&frame, 0, sizeof(frame));
    frame.struct_size = sizeof(frame);
    frame.frame_index = 0u;
    frame.reset = ns_reset_session_start;
    frame.inputs = &color;
    frame.input_count = 1u;
    frame.output = output;
    frame.camera.struct_size = sizeof(ns_camera);

    ns_evaluate_result result;
    memset(&result, 0, sizeof(result));
    result.struct_size = sizeof(result);

    /* Missing required guide must be reported, never silently substituted. */
    ns_frame no_color = frame;
    no_color.input_count = 0u;
    ns_evaluate_result no_color_result;
    memset(&no_color_result, 0, sizeof(no_color_result));
    no_color_result.struct_size = sizeof(no_color_result);
    check(adapter->evaluate(session, &no_color, &no_color_result) == ns_status_guide_missing,
          "reports guide_missing when color is absent", no_color_result.detail);
    check(no_color_result.evaluated == 0u && no_color_result.output_fresh == 0u,
          "does not claim evaluation on a failed frame", NULL);

    /* Identity transform must reproduce the input exactly. */
    ns_value value;
    memset(&value, 0, sizeof(value));
    value.type = ns_value_int;
    value.int_value = 0; /* Identity */
    adapter->set_setting(session, "Mode", &value);

    check(adapter->evaluate(session, &frame, &result) == ns_status_ok, "evaluate succeeds", result.detail);
    check(result.evaluated == 1u, "reports evaluated", NULL);
    check(result.output_fresh == 1u, "reports fresh output", NULL);
    check(memcmp(input_pixels, output_pixels, buffer_bytes) == 0,
          "identity transform reproduces the input exactly", NULL);

    /* Invert must actually change the output, so a no-op bug cannot pass. */
    value.int_value = 1; /* Invert */
    adapter->set_setting(session, "Mode", &value);
    memset(output_pixels, 0, buffer_bytes);
    frame.frame_index = 1u;
    frame.reset = ns_reset_none;
    adapter->evaluate(session, &frame, &result);
    check(memcmp(input_pixels, output_pixels, buffer_bytes) != 0,
          "invert transform changes the output", NULL);

    printf("\n[settings]\n");
    ns_value read_back;
    memset(&read_back, 0, sizeof(read_back));
    check(adapter->get_setting(session, "Mode", &read_back) == ns_status_ok && read_back.int_value == 1,
          "setting round-trips", NULL);

    value.type = ns_value_float;
    value.float_value = 5.0; /* outside the declared 0..1 range */
    check(adapter->set_setting(session, "Strength", &value) != ns_status_ok,
          "rejects an out-of-range setting value", NULL);

    check(adapter->set_setting(session, "NoSuchSetting", &value) == ns_status_unsupported,
          "reports unsupported for an unknown setting key", NULL);

    printf("\n[failure isolation]\n");
    value.type = ns_value_int;
    value.int_value = 0;
    adapter->set_setting(session, "SimulateFailureAfterFrames", &value);
    value.int_value = 1;
    adapter->set_setting(session, "SimulateFailureAfterFrames", &value);

    ns_evaluate_result failure_result;
    memset(&failure_result, 0, sizeof(failure_result));
    failure_result.struct_size = sizeof(failure_result);
    frame.frame_index = 2u;
    ns_status failure_status = adapter->evaluate(session, &frame, &failure_result);
    check(failure_status != ns_status_ok, "injected failure surfaces as an error", NULL);
    check(failure_result.evaluated == 0u && failure_result.output_fresh == 0u,
          "does not claim evaluation on an injected failure", NULL);
    check(failure_result.detail[0] != '\0', "explains the injected failure", NULL);

    printf("\n[teardown]\n");
    check(adapter->destroy_session(session) == ns_status_ok, "destroy_session succeeds", NULL);
    check(adapter->query_vram(NULL, NULL) == ns_status_invalid_argument,
          "rejects null arguments instead of faulting", NULL);
    adapter->shutdown();

    free(input_pixels);
    free(output_pixels);
    FreeLibrary(module);

    printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
