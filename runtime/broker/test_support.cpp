/*
 * NeuralStudio runtime - minimal test support
 * SPDX-License-Identifier: MIT
 */

#include "test_support.h"

#include <cstdio>

namespace nstest {
namespace {

int g_checks = 0;
int g_failures = 0;

}  // namespace

void Group(const char* name) {
    std::printf("\n[%s]\n", name);
}

void Check(bool condition, const std::string& name, const std::string& detail) {
    ++g_checks;
    if (condition) {
        std::printf("  PASS  %s\n", name.c_str());
    } else {
        ++g_failures;
        std::printf("  FAIL  %s%s%s\n", name.c_str(),
                    detail.empty() ? "" : " - ", detail.c_str());
    }
}

int TotalChecks() { return g_checks; }
int TotalFailures() { return g_failures; }

}  // namespace nstest
