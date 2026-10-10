#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0 || (size % 4) != 0)
    throw std::runtime_error("invalid uint32 table size: " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

void Write(const std::filesystem::path& path,
           const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest)
    text << std::setw(2) << static_cast<unsigned int>(byte);
  return text.str();
}

std::uint32_t LoadU32(const std::vector<std::uint8_t>& bytes,
                      std::size_t index) {
  std::uint32_t value = 0;
  std::memcpy(&value, bytes.data() + index * 4u, sizeof(value));
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-ex2-pack-fast-corrections <oracle-u32> <fast-u32> "
                 "<packed-output> <receipt-output>\n";
    return 2;
  }
  try {
    const auto oracle = Read(argv[1]);
    const auto fast = Read(argv[2]);
    const auto output_path = std::filesystem::path(argv[3]);
    const auto receipt_path = std::filesystem::path(argv[4]);
    if (oracle.size() != fast.size())
      throw std::runtime_error("input sizes differ");
    if (std::filesystem::exists(output_path) ||
        std::filesystem::exists(receipt_path))
      throw std::runtime_error("output exists");
    const std::size_t records = oracle.size() / 4u;
    std::vector<std::uint8_t> packed((records + 3u) / 4u, 0u);
    std::array<std::uint64_t, 4> histogram{};
    std::int64_t minimum = std::numeric_limits<std::int64_t>::max();
    std::int64_t maximum = std::numeric_limits<std::int64_t>::min();
    std::uint64_t reconstruction_mismatches = 0u;
    for (std::size_t index = 0; index < records; ++index) {
      const auto oracle_bits = LoadU32(oracle, index);
      const auto fast_bits = LoadU32(fast, index);
      const auto delta = static_cast<std::int64_t>(fast_bits) -
                         static_cast<std::int64_t>(oracle_bits);
      minimum = std::min(minimum, delta);
      maximum = std::max(maximum, delta);
      if (delta < -1 || delta > 2)
        throw std::runtime_error("correction outside two-bit domain at record " +
                                 std::to_string(index));
      const auto code = static_cast<std::uint8_t>(delta + 1);
      ++histogram[code];
      packed[index >> 2u] |=
          static_cast<std::uint8_t>(code << ((index & 3u) * 2u));
      const auto reconstructed = static_cast<std::uint32_t>(
          static_cast<std::int64_t>(fast_bits) - delta);
      reconstruction_mismatches += reconstructed != oracle_bits;
    }
    if (reconstruction_mismatches != 0u)
      throw std::runtime_error("packed reconstruction mismatch");
    Write(output_path, packed);
    std::ofstream receipt(receipt_path, std::ios::binary | std::ios::trunc);
    if (!receipt) throw std::runtime_error("unable to write receipt");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.sm120-ex2-fast-corrections.v1\",\n"
            << "  \"records\": " << records << ",\n"
            << "  \"oracle_sha256\": \"" << Sha256(oracle) << "\",\n"
            << "  \"fast_sha256\": \"" << Sha256(fast) << "\",\n"
            << "  \"packed_bytes\": " << packed.size() << ",\n"
            << "  \"packed_sha256\": \"" << Sha256(packed) << "\",\n"
            << "  \"delta_min\": " << minimum << ",\n"
            << "  \"delta_max\": " << maximum << ",\n"
            << "  \"delta_histogram\": {\"-1\": " << histogram[0]
            << ", \"0\": " << histogram[1] << ", \"1\": "
            << histogram[2] << ", \"2\": " << histogram[3] << "},\n"
            << "  \"reconstruction_mismatches\": "
            << reconstruction_mismatches << ",\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"pass\"\n"
            << "}\n";
    std::cout << "records=" << records << '\n'
              << "packed_bytes=" << packed.size() << '\n'
              << "packed_sha256=" << Sha256(packed) << '\n'
              << "delta_min=" << minimum << '\n'
              << "delta_max=" << maximum << '\n'
              << "result=pass\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
