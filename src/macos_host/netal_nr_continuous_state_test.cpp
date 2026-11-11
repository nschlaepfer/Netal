#include "netal_nr_continuous_state.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using netal_nr_adapter::AdapterMode;
using netal_nr_adapter::ContinuousAdapterConfig;
using netal_nr_adapter::ContinuousAdapterState;
using netal_nr_adapter::NrProfile;
using Action = ContinuousAdapterState::Action;

static ContinuousAdapterConfig parse_map(const std::map<std::string, std::string>& env, bool expect_ok = true) {
  ContinuousAdapterConfig cfg;
  const bool ok = ContinuousAdapterConfig::parse_from_lookup(
      [&](const char* key) -> const char* {
        auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
      },
      cfg);
  assert(ok == expect_ok);
  assert(cfg.valid == expect_ok);
  return cfg;
}

int main() {
  // 1. Default unset environment stays strictly Off.
  {
    auto cfg = parse_map({});
    assert(cfg.mode == AdapterMode::Off);
    assert(cfg.scale == 1.0f);
    assert(cfg.profile == NrProfile::Standard);
    assert(!cfg.async_worker && !cfg.hook_spatial && !cfg.hook_drawable);
    ContinuousAdapterState st(cfg);
    auto d = st.begin_frame(1, 7, 1920, 1080, true);
    assert(d.action == Action::Disabled && !d.admitted);
    assert(st.snapshot().telemetry.admitted == 0);
  }

  // 2. Valid parsing across all modes, profiles, ranges, and multiple-of-64 >= 128 network sizing.
  {
    auto cfg = parse_map({
      {"NETAL_NR_ADAPTER_MODE", "continuous"},
      {"NETAL_NR_SCALE", "0.5"},
      {"NETAL_NR_PROFILE", "cinematic"},
      {"NETAL_NR_DETAIL", "4.5"},
      {"NETAL_NR_COLOUR", "2.25"},
      {"NETAL_NR_INTENSITY", "0.85"},
      {"NETAL_NR_ASYNC", "1"},
      {"NETAL_NR_HOOK_SPATIAL", "1"},
      {"NETAL_NR_HOOK_DRAWABLE", "1"},
    });
    assert(cfg.mode == AdapterMode::Continuous);
    assert(cfg.profile == NrProfile::Cinematic);
    assert(cfg.async_worker && cfg.hook_spatial && cfg.hook_drawable);

    uint32_t nw = 0, nh = 0;
    assert(cfg.compute_network_extent(1920, 1080, nw, nh));
    assert(nw == 960 && nh == 512 && (nw % 64) == 0 && (nh % 64) == 0);

    // Minimum clamp >= 128 even at scale=0.25 on small valid target.
    auto min_cfg = parse_map({{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_SCALE", "0.25"}});
    assert(min_cfg.compute_network_extent(256, 256, nw, nh));
    assert(nw == 128 && nh == 128);
    assert(!min_cfg.compute_network_extent(64, 64, nw, nh));
  }

  // 3. Fail-closed validation on invalid/nonfinite/out-of-range environment parameters.
  {
    const std::vector<std::map<std::string, std::string>> bad_envs = {
      {{"NETAL_NR_ADAPTER_MODE", "turbo"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_SCALE", "0.24"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_SCALE", "1.01"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_SCALE", "nan"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_SCALE", "inf"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_SCALE", "0.5px"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_PROFILE", "vivid"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_DETAIL", "-0.1"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_DETAIL", "8.1"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_COLOUR", "4.5"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_INTENSITY", "1.2"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_INTENSITY", "-inf"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_ASYNC", "true"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_HOOK_SPATIAL", "2"}},
      {{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_HOOK_DRAWABLE", "yes"}},
    };
    for (const auto& env : bad_envs) {
      auto bad_cfg = parse_map(env, false);
      assert(bad_cfg.mode == AdapterMode::Off);
      ContinuousAdapterState st(bad_cfg);
      auto d = st.begin_frame(1, 1, 1920, 1080, true);
      assert(d.action == Action::Disabled && !d.admitted);
    }
  }

  // 4. Synchronous inline continuous mode: stream isolation, motion synthesis counter, resize -> reset, busy handling.
  {
    auto cfg = parse_map({{"NETAL_NR_ADAPTER_MODE", "continuous"}, {"NETAL_NR_SCALE", "1.0"}});
    ContinuousAdapterState st(cfg);

    // First frame on stream 10 without native motion -> admitted, reset=true, synthesize_motion=true.
    auto d1 = st.begin_frame(100, 10, 1920, 1080, false);
    assert(d1.action == Action::AdmitInline && d1.admitted && d1.slot == 0);
    assert(d1.reset && d1.synthesize_motion);
    assert(d1.network_width == 1920 && d1.network_height == 1088);

    // Other stream is ignored without perturbing state.
    auto d_other = st.begin_frame(101, 99, 1280, 720, true);
    assert(d_other.action == Action::OtherStream && !d_other.admitted);

    // Overlapping frame before slot 0 retires -> BusyPassthrough, forces reset on next admitted frame.
    auto d_busy = st.begin_frame(101, 10, 1920, 1080, true);
    assert(d_busy.action == Action::BusyPassthrough && !d_busy.admitted);

    // Complete frame 100 and publish.
    assert(!st.complete_slot(0, d1.generation, 999, true)); // wrong frame_id rejected
    assert(st.complete_slot(0, d1.generation, 100, true));
    assert(st.note_published(10, d1.generation));

    // Next admitted frame has reset=true because frame 101 missed inline continuity!
    auto d2 = st.begin_frame(102, 10, 1920, 1080, true);
    assert(d2.action == Action::AdmitInline && d2.reset && !d2.synthesize_motion);
    assert(st.complete_slot(0, d2.generation, 102, true));
    assert(st.note_published(10, d2.generation));

    // Contiguous frame 103 has reset=false!
    auto d3 = st.begin_frame(103, 10, 1920, 1080, true);
    assert(d3.action == Action::AdmitInline && !d3.reset);
    assert(st.complete_slot(0, d3.generation, 103, true));
    assert(st.note_published(10, d3.generation));

    // Resize 1920x1080 -> 2560x1440 triggers automatic history reset=true and new network extent!
    auto d4 = st.begin_frame(104, 10, 2560, 1440, false);
    assert(d4.action == Action::AdmitInline && d4.reset && d4.synthesize_motion);
    assert(d4.network_width == 2560 && d4.network_height == 1472);
    assert(st.complete_slot(0, d4.generation, 104, true));

    auto snap = st.snapshot();
    assert(snap.telemetry.admitted == 4);
    assert(snap.telemetry.completed == 4);
    assert(snap.telemetry.published == 3);
    assert(snap.telemetry.synthesized_motion_frames == 2);
    assert(snap.telemetry.resets == 3);
    assert(snap.telemetry.skipped_busy == 1);
  }

  // 5. Double-buffered async worker mode: never stalls, reuses latest completed delta while worker is busy,
  //    invalidates stale deltas on resize or pause control file!
  {
    auto cfg = parse_map({
      {"NETAL_NR_ADAPTER_MODE", "continuous"},
      {"NETAL_NR_SCALE", "0.75"},
      {"NETAL_NR_ASYNC", "1"},
    });
    ContinuousAdapterState st(cfg);

    // Frame 1: admitted on slot 0, no prior delta yet.
    auto f1 = st.begin_frame(1, 5, 1920, 1080, false);
    assert(f1.action == Action::AdmitAsync && f1.admitted && f1.slot == 0 && f1.reset && !f1.has_reusable_delta);

    // Frame 2 arrives while slot 0 is still computing -> BusyPassthrough (no delta ready yet), zero stall!
    auto f2 = st.begin_frame(2, 5, 1920, 1080, false);
    assert(f2.action == Action::BusyPassthrough && !f2.admitted);

    // Slot 0 finishes!
    assert(st.complete_slot(0, f1.generation, 1, true));

    // Frame 3 arrives: slot 0 has valid delta (frame 1), so slot 1 is reserved for new async compute
    // while frame 3 ALSO receives slot 0 as reusable delta!
    auto f3 = st.begin_frame(3, 5, 1920, 1080, false);
    assert(f3.action == Action::AdmitAsync && f3.admitted && f3.slot == 1 && !f3.reset);
    assert(f3.has_reusable_delta && f3.reusable_slot == 0 && f3.reusable_frame == 1);
    assert(st.note_published(5, f3.generation));

    // Frame 4 arrives while slot 1 is STILL busy -> ReuseLatestDelta from slot 0 (never stalls!).
    auto f4 = st.begin_frame(4, 5, 1920, 1080, false);
    assert(f4.action == Action::ReuseLatestDelta && !f4.admitted);
    assert(f4.has_reusable_delta && f4.reusable_slot == 0 && f4.reusable_frame == 1);
    assert(st.note_published(5, f4.generation));

    // Slot 1 completes -> latest_ready_slot switches to slot 1!
    assert(st.complete_slot(1, f3.generation, 3, true));

    // Pause control file round-trip.
    char tmpl[] = "/tmp/netal-nr-pause-XXXXXX";
    int fd = ::mkstemp(tmpl);
    assert(fd >= 0);
    ::close(fd);
    std::string pause_file(tmpl);

    assert(st.poll_pause_file(pause_file));
    auto fpaused = st.begin_frame(5, 5, 1920, 1080, false);
    assert(fpaused.action == Action::Paused && !fpaused.admitted);

    ::unlink(pause_file.c_str());
    assert(!st.poll_pause_file(pause_file));

    // Resuming after pause forces reset=true and clears stale pre-pause delta!
    auto f6 = st.begin_frame(6, 5, 1920, 1080, false);
    assert(f6.action == Action::AdmitAsync && f6.reset && !f6.has_reusable_delta);
    assert(st.complete_slot(f6.slot, f6.generation, 6, true));

    auto snap = st.snapshot();
    assert(snap.telemetry.admitted == 3);
    assert(snap.telemetry.completed == 3);
    assert(snap.telemetry.published == 2);
    assert(snap.telemetry.synthesized_motion_frames == 3);
    assert(snap.telemetry.resets == 2);
    assert(snap.telemetry.skipped_busy == 2);
  }

  // 5b. Async worker cadence throttle (NETAL_NR_WORKER_INTERVAL_MS).
  {
    auto bad_iv = parse_map(
        {
            {"NETAL_NR_ADAPTER_MODE", "continuous"},
            {"NETAL_NR_WORKER_INTERVAL_MS", "1500"},
        },
        false);
    assert(!bad_iv.valid);

    auto cfg = parse_map({
        {"NETAL_NR_ADAPTER_MODE", "continuous"},
        {"NETAL_NR_ASYNC", "1"},
        {"NETAL_NR_WORKER_INTERVAL_MS", "200"},
    });
    assert(cfg.valid);
    assert(cfg.worker_interval_ms == 200);
    ContinuousAdapterState st(cfg);

    auto f1 = st.begin_frame(1, 1, 1920, 1080, false);
    assert(f1.action == ContinuousAdapterState::Action::AdmitAsync);
    st.complete_slot(f1.slot, f1.generation, 1, true);

    // Immediate next frame within 200 ms window reuses cached delta without admitting worker.
    auto f2 = st.begin_frame(2, 1, 1920, 1080, false);
    assert(f2.action == ContinuousAdapterState::Action::ReuseLatestDelta);
    assert(!f2.admitted);
    assert(f2.has_reusable_delta);

    // Explicit game_reset bypasses throttle immediately.
    auto f3 = st.begin_frame(3, 1, 1920, 1080, false, true);
    assert(f3.action == ContinuousAdapterState::Action::AdmitAsync);
    assert(f3.admitted);
    st.complete_slot(f3.slot, f3.generation, 3, true);
  }

  // 6. Thread-safety stress test on concurrent admission/completion.
  {
    auto cfg = parse_map({{"NETAL_NR_ADAPTER_MODE", "once"}});
    ContinuousAdapterState st(cfg);
    std::atomic<unsigned> winners{0};
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < 16; ++i) {
      threads.emplace_back([&, i] {
        auto d = st.begin_frame(10 + i, 3, 1920, 1080, true);
        if (d.admitted) ++winners;
      });
    }
    for (auto& t : threads) t.join();
    assert(winners.load() == 1);
  }

  // 7. Direct parse_from_env smoke check.
  {
    ::unsetenv("NETAL_NR_ADAPTER_MODE");
    ContinuousAdapterConfig env_cfg;
    assert(ContinuousAdapterConfig::parse_from_env(env_cfg));
    assert(env_cfg.mode == AdapterMode::Off);
  }

  std::puts("PASS continuous NR adapter config, fail-closed validation, resize reset, pause control, and async double-buffering tests");
  return 0;
}
