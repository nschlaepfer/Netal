#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kChunkRecords = 1u << 20u;

std::uint64_t FileSize(const std::filesystem::path& path) {
  const auto bytes = std::filesystem::file_size(path);
  if ((bytes % sizeof(std::uint32_t)) != 0u) {
    throw std::runtime_error("file size is not uint32 aligned: " +
                             path.string());
  }
  return bytes;
}

void ReadExact(std::ifstream& stream, std::vector<std::uint32_t>& output,
               std::size_t records, const std::filesystem::path& path) {
  const auto bytes = static_cast<std::streamsize>(records * sizeof(std::uint32_t));
  stream.read(reinterpret_cast<char*>(output.data()), bytes);
  if (stream.gcount() != bytes) {
    throw std::runtime_error("short read: " + path.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage=fg-ex2-analyze-fast-deltas <oracle-u32> <fast-u32>\n";
    return 2;
  }

  try {
    const auto oracle_path = std::filesystem::path(argv[1]);
    const auto fast_path = std::filesystem::path(argv[2]);
    const auto oracle_bytes = FileSize(oracle_path);
    const auto fast_bytes = FileSize(fast_path);
    if (oracle_bytes != fast_bytes) {
      throw std::runtime_error("input sizes differ");
    }

    std::ifstream oracle_stream(oracle_path, std::ios::binary);
    std::ifstream fast_stream(fast_path, std::ios::binary);
    if (!oracle_stream || !fast_stream) {
      throw std::runtime_error("unable to open input table");
    }

    std::vector<std::uint32_t> oracle(kChunkRecords);
    std::vector<std::uint32_t> fast(kChunkRecords);
    const std::uint64_t record_count = oracle_bytes / sizeof(std::uint32_t);
    std::map<std::int64_t, std::uint64_t> histogram;
    std::uint64_t mismatched_records = 0u;
    std::uint64_t first_mismatch = std::numeric_limits<std::uint64_t>::max();
    std::uint32_t first_oracle = 0u;
    std::uint32_t first_fast = 0u;

    for (std::uint64_t base = 0u; base < record_count;
         base += kChunkRecords) {
      const auto records = static_cast<std::size_t>(
          std::min<std::uint64_t>(kChunkRecords, record_count - base));
      ReadExact(oracle_stream, oracle, records, oracle_path);
      ReadExact(fast_stream, fast, records, fast_path);
      for (std::size_t index = 0u; index < records; ++index) {
        const auto delta = static_cast<std::int64_t>(fast[index]) -
                           static_cast<std::int64_t>(oracle[index]);
        ++histogram[delta];
        if (delta != 0) {
          ++mismatched_records;
          if (first_mismatch == std::numeric_limits<std::uint64_t>::max()) {
            first_mismatch = base + index;
            first_oracle = oracle[index];
            first_fast = fast[index];
          }
        }
      }
    }

    std::cout << "records=" << record_count << '\n'
              << "mismatched_records=" << mismatched_records << '\n'
              << "mismatch_fraction=" << std::setprecision(12)
              << static_cast<double>(mismatched_records) /
                     static_cast<double>(record_count)
              << '\n'
              << "delta_min=" << histogram.begin()->first << '\n'
              << "delta_max=" << histogram.rbegin()->first << '\n';
    if (mismatched_records != 0u) {
      std::cout << "first_mismatch_index=" << first_mismatch << '\n'
                << "first_oracle_bits=0x" << std::hex << std::setw(8)
                << std::setfill('0') << first_oracle << '\n'
                << "first_fast_bits=0x" << std::setw(8) << first_fast
                << std::dec << '\n';
    }
    std::cout << "delta_histogram_begin\n";
    for (const auto& [delta, count] : histogram) {
      std::cout << delta << ' ' << count << '\n';
    }
    std::cout << "delta_histogram_end\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
