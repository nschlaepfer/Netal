#pragma once
// CPU-side admission, configuration, geometry-reset, and async double-buffered slot
// state machine for Netal's NR adapter. Pure C++20; does not execute GPU work itself.
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unistd.h>

namespace netal_nr_adapter {

enum class AdapterMode { Off, Once, DelayedOnce, Continuous };
enum class NrProfile { Standard, Natural, Cinematic, Neutral };

inline const char* AdapterModeName(AdapterMode m) {
  switch (m) {
    case AdapterMode::Off: return "off";
    case AdapterMode::Once: return "once";
    case AdapterMode::DelayedOnce: return "delayed_once";
    case AdapterMode::Continuous: return "continuous";
  }
  return "off";
}

inline const char* NrProfileName(NrProfile p) {
  switch (p) {
    case NrProfile::Standard: return "standard";
    case NrProfile::Natural: return "natural";
    case NrProfile::Cinematic: return "cinematic";
    case NrProfile::Neutral: return "neutral";
  }
  return "standard";
}

struct ContinuousAdapterConfig {
  AdapterMode mode = AdapterMode::Off;
  float scale = 1.0f;           // [0.25, 1.0]
  NrProfile profile = NrProfile::Standard;
  float detail = 1.0f;          // [0.0, 8.0]
  float colour = 1.0f;          // [0.0, 4.0]
  float intensity = 1.0f;       // [0.0, 1.0]
  bool async_worker = false;    // 0 | 1
  uint32_t worker_interval_ms = 0; // [0, 1000] minimum ms between async worker completions/admissions
  bool hook_spatial = false;    // 0 | 1
  bool hook_drawable = false;   // 0 | 1
  bool valid = true;
  std::string error;

  using EnvLookup = std::function<const char*(const char*)>;

  static std::string_view trim_ascii(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
      s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
      s.remove_suffix(1);
    }
    return s;
  }

  static bool parse_float_in_range(const char* raw, float min_v, float max_v, float& out) {
    std::string_view sv = trim_ascii(raw ? std::string_view(raw) : std::string_view{});
    if (sv.empty()) return true;
    std::string buf(sv);
    errno = 0;
    char* end = nullptr;
    const float val = std::strtof(buf.c_str(), &end);
    if (errno == ERANGE || end == buf.c_str() || *end != '\0' || !std::isfinite(val)) return false;
    if (val < min_v || val > max_v) return false;
    out = val;
    return true;
  }

  static bool parse_bool01(const char* raw, bool& out) {
    std::string_view sv = trim_ascii(raw ? std::string_view(raw) : std::string_view{});
    if (sv.empty()) return true;
    if (sv == "0") { out = false; return true; }
    if (sv == "1") { out = true; return true; }
    return false;
  }

  static bool parse_from_lookup(const EnvLookup& lookup, ContinuousAdapterConfig& out) {
    out = ContinuousAdapterConfig{};
    auto fail = [&](const char* msg) -> bool {
      out = ContinuousAdapterConfig{};
      out.valid = false;
      out.mode = AdapterMode::Off;
      out.error = msg;
      return false;
    };

    if (const char* raw_mode = lookup("NETAL_NR_ADAPTER_MODE")) {
      std::string_view m = trim_ascii(raw_mode);
      if (m.empty() || m == "off") out.mode = AdapterMode::Off;
      else if (m == "once") out.mode = AdapterMode::Once;
      else if (m == "delayed_once") out.mode = AdapterMode::DelayedOnce;
      else if (m == "continuous") out.mode = AdapterMode::Continuous;
      else return fail("invalid NETAL_NR_ADAPTER_MODE (expected off|once|delayed_once|continuous)");
    }

    if (!parse_float_in_range(lookup("NETAL_NR_SCALE"), 0.25f, 1.0f, out.scale)) {
      return fail("invalid NETAL_NR_SCALE (expected finite float in [0.25, 1.0])");
    }

    if (const char* raw_prof = lookup("NETAL_NR_PROFILE")) {
      std::string_view p = trim_ascii(raw_prof);
      if (p.empty() || p == "standard") out.profile = NrProfile::Standard;
      else if (p == "natural") out.profile = NrProfile::Natural;
      else if (p == "cinematic") out.profile = NrProfile::Cinematic;
      else if (p == "neutral") out.profile = NrProfile::Neutral;
      else return fail("invalid NETAL_NR_PROFILE (expected standard|natural|cinematic|neutral)");
    }

    if (!parse_float_in_range(lookup("NETAL_NR_DETAIL"), 0.0f, 8.0f, out.detail)) {
      return fail("invalid NETAL_NR_DETAIL (expected finite float in [0, 8])");
    }
    if (!parse_float_in_range(lookup("NETAL_NR_COLOUR"), 0.0f, 4.0f, out.colour)) {
      return fail("invalid NETAL_NR_COLOUR (expected finite float in [0, 4])");
    }
    if (!parse_float_in_range(lookup("NETAL_NR_INTENSITY"), 0.0f, 1.0f, out.intensity)) {
      return fail("invalid NETAL_NR_INTENSITY (expected finite float in [0, 1])");
    }
    if (!parse_bool01(lookup("NETAL_NR_ASYNC"), out.async_worker)) {
      return fail("invalid NETAL_NR_ASYNC (expected 0 or 1)");
    }
    if (const char* raw_iv = lookup("NETAL_NR_WORKER_INTERVAL_MS")) {
      float iv_f = 0.0f;
      if (!parse_float_in_range(raw_iv, 0.0f, 1000.0f, iv_f)) {
        return fail("invalid NETAL_NR_WORKER_INTERVAL_MS (expected number in [0, 1000])");
      }
      out.worker_interval_ms = static_cast<uint32_t>(std::llround(iv_f));
    }
    if (!parse_bool01(lookup("NETAL_NR_HOOK_SPATIAL"), out.hook_spatial)) {
      return fail("invalid NETAL_NR_HOOK_SPATIAL (expected 0 or 1)");
    }
    if (!parse_bool01(lookup("NETAL_NR_HOOK_DRAWABLE"), out.hook_drawable)) {
      return fail("invalid NETAL_NR_HOOK_DRAWABLE (expected 0 or 1)");
    }
    return true;
  }

  static bool parse_from_env(ContinuousAdapterConfig& out) {
    return parse_from_lookup([](const char* key) { return std::getenv(key); }, out);
  }

  bool validate_self() const {
    if (!valid) return false;
    if (!std::isfinite(scale) || scale < 0.25f || scale > 1.0f) return false;
    if (!std::isfinite(detail) || detail < 0.0f || detail > 8.0f) return false;
    if (!std::isfinite(colour) || colour < 0.0f || colour > 4.0f) return false;
    if (!std::isfinite(intensity) || intensity < 0.0f || intensity > 1.0f) return false;
    if (worker_interval_ms > 1000u) return false;
    return true;
  }

  bool compute_network_extent(uint32_t output_width, uint32_t output_height,
                              uint32_t& network_width, uint32_t& network_height) const {
    network_width = 0;
    network_height = 0;
    if (!validate_self()) return false;
    if (output_width < 128u || output_height < 128u ||
        output_width > 16384u || output_height > 16384u) {
      return false;
    }
    auto round64 = [](uint32_t dim, float s) -> uint32_t {
      const double scaled = static_cast<double>(dim) * static_cast<double>(s);
      const uint32_t rounded = static_cast<uint32_t>(std::llround(scaled / 64.0)) * 64u;
      return std::clamp(rounded, 128u, 16384u);
    };
    network_width = round64(output_width, scale);
    network_height = round64(output_height, scale);
    return (network_width % 64u) == 0u && (network_height % 64u) == 0u &&
           network_width >= 128u && network_height >= 128u;
  }
};

class ContinuousAdapterState {
public:
  static constexpr uint32_t kNumSlots = 2;

  enum class Action {
    Disabled,
    Paused,
    InvalidGeometry,
    OtherStream,
    OneShotExhausted,
    BusyPassthrough,
    ReuseLatestDelta,
    AdmitInline,
    AdmitAsync,
  };

  struct Telemetry {
    uint64_t admitted = 0;
    uint64_t completed = 0;
    uint64_t published = 0;
    uint64_t synthesized_motion_frames = 0;
    uint64_t resets = 0;
    uint64_t skipped_busy = 0;
  };

  struct SlotInfo {
    bool in_flight = false;
    bool valid_delta = false;
    uint64_t frame_id = 0;
    uint64_t generation = 0;
  };

  struct Snapshot {
    ContinuousAdapterConfig config{};
    bool active = false;
    bool paused = false;
    uint64_t selected_stream = 0;
    uint64_t generation = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    uint32_t network_width = 0;
    uint32_t network_height = 0;
    bool force_reset_next = true;
    int32_t latest_ready_slot = -1;
    std::array<SlotInfo, kNumSlots> slots{};
    Telemetry telemetry{};
  };

  struct FrameDecision {
    Action action = Action::Disabled;
    bool admitted = false;
    uint32_t slot = 0;
    uint64_t generation = 0;
    bool reset = false;
    bool synthesize_motion = false;
    uint32_t network_width = 0;
    uint32_t network_height = 0;
    bool has_reusable_delta = false;
    uint32_t reusable_slot = 0;
    uint64_t reusable_frame = 0;
  };

private:
  mutable std::mutex mutex_;
  ContinuousAdapterConfig config_{};
  bool paused_ = false;
  uint64_t selected_stream_ = 0;
  uint64_t generation_ = 1;
  uint32_t output_width_ = 0;
  uint32_t output_height_ = 0;
  uint32_t network_width_ = 0;
  uint32_t network_height_ = 0;
  bool force_reset_next_ = true;
  bool oneshot_consumed_ = false;
  bool delayed_once_published_ = false;
  int32_t latest_ready_slot_ = -1;
  bool has_last_complete_time_ = false;
  std::chrono::steady_clock::time_point last_complete_time_{};
  std::array<SlotInfo, kNumSlots> slots_{};
  Telemetry telemetry_{};

  bool is_enabled_locked() const {
    return config_.validate_self() && config_.mode != AdapterMode::Off;
  }

  void invalidate_deltas_locked() {
    latest_ready_slot_ = -1;
    has_last_complete_time_ = false;
    for (auto& s : slots_) {
      s.valid_delta = false;
    }
    ++generation_;
    force_reset_next_ = true;
  }

public:
  explicit ContinuousAdapterState(const ContinuousAdapterConfig& cfg = ContinuousAdapterConfig{}) {
    configure(cfg);
  }

  bool configure(const ContinuousAdapterConfig& cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cfg.validate_self()) {
      config_ = ContinuousAdapterConfig{};
      config_.valid = false;
      config_.mode = AdapterMode::Off;
      config_.error = cfg.error.empty() ? "invalid ContinuousAdapterConfig" : cfg.error;
      invalidate_deltas_locked();
      return false;
    }
    config_ = cfg;
    selected_stream_ = 0;
    output_width_ = 0;
    output_height_ = 0;
    network_width_ = 0;
    network_height_ = 0;
    oneshot_consumed_ = false;
    delayed_once_published_ = false;
    for (auto& s : slots_) s = SlotInfo{};
    invalidate_deltas_locked();
    return true;
  }

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot snap{};
    snap.config = config_;
    snap.active = is_enabled_locked() && !paused_;
    snap.paused = paused_;
    snap.selected_stream = selected_stream_;
    snap.generation = generation_;
    snap.output_width = output_width_;
    snap.output_height = output_height_;
    snap.network_width = network_width_;
    snap.network_height = network_height_;
    snap.force_reset_next = force_reset_next_;
    snap.latest_ready_slot = latest_ready_slot_;
    snap.slots = slots_;
    snap.telemetry = telemetry_;
    return snap;
  }

  void set_paused(bool paused) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (paused_ != paused) {
      paused_ = paused;
      // Temporal continuity is broken across unmonitored paused game frames.
      invalidate_deltas_locked();
    }
  }

  bool poll_pause_file(const std::string& pause_marker_path) {
    const bool exists = !pause_marker_path.empty() && ::access(pause_marker_path.c_str(), F_OK) == 0;
    set_paused(exists);
    return exists;
  }

  void force_reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    invalidate_deltas_locked();
  }

  void reselect_stream() {
    std::lock_guard<std::mutex> lock(mutex_);
    selected_stream_ = 0;
    output_width_ = 0;
    output_height_ = 0;
    network_width_ = 0;
    network_height_ = 0;
    invalidate_deltas_locked();
  }

  FrameDecision begin_frame(uint64_t frame_id, uint64_t stream_id,
                            uint32_t output_width, uint32_t output_height,
                            bool has_native_motion, bool game_reset = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    FrameDecision d{};
    if (!is_enabled_locked()) {
      d.action = Action::Disabled;
      return d;
    }
    if (paused_) {
      d.action = Action::Paused;
      return d;
    }
    if (stream_id == 0) {
      d.action = Action::InvalidGeometry;
      return d;
    }
    uint32_t net_w = 0, net_h = 0;
    if (!config_.compute_network_extent(output_width, output_height, net_w, net_h)) {
      if (selected_stream_ == stream_id) invalidate_deltas_locked();
      d.action = Action::InvalidGeometry;
      return d;
    }

    if (selected_stream_ == 0) {
      selected_stream_ = stream_id;
      output_width_ = output_width;
      output_height_ = output_height;
      network_width_ = net_w;
      network_height_ = net_h;
      force_reset_next_ = true;
    } else if (stream_id != selected_stream_) {
      d.action = Action::OtherStream;
      return d;
    } else if (output_width != output_width_ || output_height != output_height_) {
      output_width_ = output_width;
      output_height_ = output_height;
      network_width_ = net_w;
      network_height_ = net_h;
      invalidate_deltas_locked();
    }

    d.network_width = network_width_;
    d.network_height = network_height_;
    d.generation = generation_;

    // Populate currently available completed async delta (if valid for current generation).
    if (latest_ready_slot_ >= 0) {
      const uint32_t rslot = static_cast<uint32_t>(latest_ready_slot_);
      const auto& ready_info = slots_[rslot];
      if (!ready_info.in_flight && ready_info.valid_delta && ready_info.generation == generation_) {
        d.has_reusable_delta = true;
        d.reusable_slot = rslot;
        d.reusable_frame = ready_info.frame_id;
      }
    }

    // One-shot / Delayed-once lifecycle gates.
    if (config_.mode == AdapterMode::Once && oneshot_consumed_) {
      d.action = Action::OneShotExhausted;
      return d;
    }
    if (config_.mode == AdapterMode::DelayedOnce) {
      if (delayed_once_published_) {
        d.action = Action::OneShotExhausted;
        return d;
      }
      if (oneshot_consumed_) {
        if (d.has_reusable_delta) {
          delayed_once_published_ = true;
          d.action = Action::ReuseLatestDelta;
        } else {
          ++telemetry_.skipped_busy;
          d.action = Action::BusyPassthrough;
        }
        return d;
      }
    }

    // Synchronous inline mode: single active slot (slot 0); never stalls if prior CB hasn't retired.
    if (!config_.async_worker && config_.mode != AdapterMode::DelayedOnce) {
      if (slots_[0].in_flight) {
        ++telemetry_.skipped_busy;
        force_reset_next_ = true;
        d.action = Action::BusyPassthrough;
        return d;
      }
      const bool do_reset = force_reset_next_ || game_reset;
      force_reset_next_ = false;
      slots_[0].in_flight = true;
      slots_[0].valid_delta = false;
      slots_[0].frame_id = frame_id;
      slots_[0].generation = generation_;
      oneshot_consumed_ = true;

      ++telemetry_.admitted;
      if (do_reset) ++telemetry_.resets;
      if (!has_native_motion) ++telemetry_.synthesized_motion_frames;

      d.action = Action::AdmitInline;
      d.admitted = true;
      d.slot = 0;
      d.reset = do_reset;
      d.synthesize_motion = !has_native_motion;
      return d;
    }

    // Double-buffered async worker mode (and DelayedOnce):
    // Keep latest_ready_slot_ intact for lock-free display compositing while reserving the other slot.
    int32_t free_slot = -1;
    bool any_in_flight = false;
    for (uint32_t i = 0; i < kNumSlots; ++i) {
      if (slots_[i].in_flight) any_in_flight = true;
    }
    if (!any_in_flight) {
      for (uint32_t i = 0; i < kNumSlots; ++i) {
        if (static_cast<int32_t>(i) != latest_ready_slot_) {
          free_slot = static_cast<int32_t>(i);
          break;
        }
      }
      if (free_slot < 0) free_slot = 0;
    }

    if (free_slot < 0) {
      // Worker is still busy on its single active job: never block the game!
      ++telemetry_.skipped_busy;
      d.action = d.has_reusable_delta ? Action::ReuseLatestDelta : Action::BusyPassthrough;
      return d;
    }

    // Optional worker cadence throttle (NETAL_NR_WORKER_INTERVAL_MS): once at least one
    // delta is ready, leave uncontended GPU headroom between background neural updates.
    if (config_.mode == AdapterMode::Continuous && config_.worker_interval_ms > 0u &&
        d.has_reusable_delta && has_last_complete_time_ && !force_reset_next_ && !game_reset) {
      const auto elapsed = std::chrono::steady_clock::now() - last_complete_time_;
      if (elapsed < std::chrono::milliseconds(config_.worker_interval_ms)) {
        ++telemetry_.skipped_busy;
        d.action = Action::ReuseLatestDelta;
        return d;
      }
    }

    const uint32_t chosen = static_cast<uint32_t>(free_slot);
    const bool do_reset = force_reset_next_ || game_reset;
    force_reset_next_ = false;
    slots_[chosen].in_flight = true;
    slots_[chosen].valid_delta = false;
    slots_[chosen].frame_id = frame_id;
    slots_[chosen].generation = generation_;
    oneshot_consumed_ = true;

    ++telemetry_.admitted;
    if (do_reset) ++telemetry_.resets;
    if (!has_native_motion) ++telemetry_.synthesized_motion_frames;

    d.action = Action::AdmitAsync;
    d.admitted = true;
    d.slot = chosen;
    d.reset = do_reset;
    d.synthesize_motion = !has_native_motion;
    return d;
  }

  bool complete_slot(uint32_t slot, uint64_t generation, uint64_t frame_id, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= kNumSlots) return false;
    auto& s = slots_[slot];
    if (!s.in_flight || s.frame_id != frame_id) return false;
    s.in_flight = false;
    if (!success) {
      s.valid_delta = false;
      force_reset_next_ = true;
      return false;
    }
    ++telemetry_.completed;
    if (generation == generation_ && !paused_ && is_enabled_locked()) {
      s.valid_delta = true;
      latest_ready_slot_ = static_cast<int32_t>(slot);
      last_complete_time_ = std::chrono::steady_clock::now();
      has_last_complete_time_ = true;
    } else {
      s.valid_delta = false;
    }
    return true;
  }

  bool note_published(uint64_t stream_id, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_enabled_locked() || paused_ || stream_id != selected_stream_ || generation != generation_) {
      return false;
    }
    ++telemetry_.published;
    return true;
  }
};

} // namespace netal_nr_adapter
