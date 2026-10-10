// Test-only bridge: record routing, without Metal, Wine, games, or model execution.
#include <cstdio>
#include <cstdlib>
static const char* Value(const char* name) {const char* value=std::getenv(name);return value?value:"<unset>";}
__attribute__((constructor)) static void Loaded() {
  const char* path=std::getenv("NETAL_TEST_RECEIPT");if (!path) return;
  if (FILE* file=std::fopen(path,"w")) {
    std::fprintf(file,"target=%s\nsr=%s\narm_sr=%s\narm_fg=%s\n",Value("NEURALSTUDIO_TARGET_EXE"),Value("NEURALSTUDIO_RE9_SR_BUNDLE"),Value("NEURALSTUDIO_RE9_SR_ARM"),Value("NEURALSTUDIO_RE9_FG_ARM"));
    std::fclose(file);
  }
}
