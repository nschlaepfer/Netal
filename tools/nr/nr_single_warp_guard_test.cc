// CPU unit test for the pure host-guard predicate (no Metal, no GPU): every rejection case Codex asked for.
// Build: clang++ -std=c++20 -O0 -Wall -Wextra -Werror -I src/nr_temporal tools/nr/nr_single_warp_guard_test.cc -o OUT
#include "nr_single_warp_guard.h"
#include <cstdio>
#include <string>
static int fails = 0, total = 0;
static void expect(const char* name, const NrSingleWarpFacts& f, bool want) {
  std::string why; const bool got = nr_single_warp_facts_ok(f, why); ++total;
  const bool ok = got == want; if (!ok) ++fails;
  printf("%s %s :: got=%s want=%s why=%s\n", ok ? "PASS" : "FAIL", name, got ? "accept" : "reject", want ? "accept" : "reject", why.c_str());
}
int main() {
  expect("accept-32x1x1-width32", {32, 1024, 32, 1, 1}, true);
  // threadgroup shapes Codex named: 0, 31, 33, 64 - all must be refused on a correct 32-wide pipeline
  expect("reject-shape-w0",  {32, 1024, 0,  1, 1}, false);
  expect("reject-shape-w31", {32, 1024, 31, 1, 1}, false);
  expect("reject-shape-w33", {32, 1024, 33, 1, 1}, false);
  expect("reject-shape-w64", {32, 1024, 64, 1, 1}, false);
  // the other two extents must be exactly 1: a 32-thread group is not the same as 32x1x1
  expect("reject-shape-32x2x1", {32, 1024, 32, 2, 1}, false);
  expect("reject-shape-32x1x2", {32, 1024, 32, 1, 2}, false);
  expect("reject-shape-32x0x1", {32, 1024, 32, 0, 1}, false);
  expect("reject-shape-32x1x0", {32, 1024, 32, 1, 0}, false);
  expect("reject-shape-1x32x1", {32, 1024, 1, 32, 1}, false);
  // wrong pipeline execution width, even with a perfect 32x1x1 dispatch
  expect("reject-width-0",  {0,  1024, 32, 1, 1}, false);
  expect("reject-width-1",  {1,  1024, 32, 1, 1}, false);
  expect("reject-width-16", {16, 1024, 32, 1, 1}, false);
  expect("reject-width-31", {31, 1024, 32, 1, 1}, false);
  expect("reject-width-64", {64, 1024, 32, 1, 1}, false);
  // pipeline that cannot host 32 threads at all
  expect("reject-maxtotal-31", {32, 31, 32, 1, 1}, false);
  expect("reject-maxtotal-0",  {32, 0,  32, 1, 1}, false);
  expect("accept-maxtotal-32", {32, 32, 32, 1, 1}, true);
  printf("%d/%d cases passed\n", total - fails, total);
  return fails ? 1 : 0;
}
