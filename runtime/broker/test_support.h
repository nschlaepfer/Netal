/*
 * NeuralStudio runtime - minimal test support
 * SPDX-License-Identifier: MIT
 *
 * Deliberately tiny and dependency-free so the runtime's pure-logic tests build
 * with only the MSVC toolchain the repository already requires. If a real test
 * framework is adopted later, only this header and its implementation change.
 */

#ifndef NEURALSTUDIO_RUNTIME_TEST_SUPPORT_H
#define NEURALSTUDIO_RUNTIME_TEST_SUPPORT_H

#include <string>

namespace nstest {

void Group(const char* name);
void Check(bool condition, const std::string& name, const std::string& detail = "");

int TotalChecks();
int TotalFailures();

}  // namespace nstest

#endif  /* NEURALSTUDIO_RUNTIME_TEST_SUPPORT_H */
