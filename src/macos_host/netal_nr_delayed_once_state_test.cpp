#include "netal_nr_delayed_once_state.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>
using State = netal_nr_adapter::DelayedOnceState;
static void ready(State& s) {
  assert(s.begin_capture(12, 7, 1920, 1200));
  assert(s.capture_completed(true));
  assert(s.compute_completed(true));
}
int main() {
  {
    State s;assert(!s.release_compute(true));
    assert(s.begin_capture(12,7,1920,1200));assert(s.capture_completed(true,true));
    assert(s.snapshot().phase==State::Phase::WaitingForCompute);
    for(unsigned i=0;i<100;++i){assert(!s.release_compute(false));assert(!s.note_model_computed());
      assert(!s.compute_completed(true));assert(!s.begin_publication(13+i,7,1920,1200));}
    assert(s.release_compute(true));assert(!s.release_compute(true));assert(s.compute_completed(true));
    for(unsigned i=0;i<100;++i){assert(!s.begin_publication(13+i,7,1920,1200,false));
      assert(s.snapshot().phase==State::Phase::Ready && !s.snapshot().publication_may_be_encoded);}
    assert(s.begin_publication(200,7,1920,1200,true));assert(s.commit_publication());
    assert(s.publication_completed(true));assert(s.original_completed(201,7,true));
  }
  for(unsigned stage=0;stage<2;++stage) {
    State s;assert(s.begin_capture(12,7,1920,1200));assert(s.capture_completed(true,true));
    if(stage){assert(s.release_compute(true));assert(s.compute_completed(true));}
    s.fail();assert(!s.release_compute(true));assert(!s.begin_publication(13,7,1920,1200,true));
    assert(!s.snapshot().publication_may_be_encoded);
  }
  {
    State s;assert(s.begin_capture(12,7,1920,1200));assert(s.capture_completed(true,true));
    assert(!s.observe_geometry(7,1920,1080));assert(!s.release_compute(true));
  }
  {
    State s;assert(s.begin_capture(12,7,1920,1200));assert(s.capture_completed(true,true));
    std::atomic<unsigned> admitted{0};std::vector<std::thread> threads;
    for(unsigned i=0;i<16;++i)threads.emplace_back([&]{if(s.release_compute(true))++admitted;});
    for(auto& t:threads)t.join();assert(admitted==1);
  }
  {
    State s;
    assert(!s.begin_capture(12, 0, 1920, 1200));
    assert(!s.begin_capture(12, 7, 1920, 1080));
    assert(!s.capture_completed(true)); assert(!s.compute_completed(true));
    ready(s);
    assert(!s.begin_capture(13, 7, 1920, 1200));
    assert(!s.begin_publication(12, 7, 1920, 1200));
    assert(!s.begin_publication(13, 8, 1920, 1200));
    assert(s.begin_publication(50, 7, 1920, 1200));
    assert(!s.begin_publication(51, 7, 1920, 1200));
    assert(!s.note_presented(50, 7));
    assert(s.commit_publication());assert(!s.commit_publication());
    assert(!s.note_presented(12, 7)); assert(!s.note_presented(50, 8));
    assert(s.note_presented(50, 7));
    assert(!s.original_completed(51, 7, true));
    assert(s.publication_completed(true));
    assert(!s.original_completed(50, 7, true)); assert(!s.original_completed(51, 8, true));
    assert(s.original_completed(51, 7, true));
    assert(!s.begin_capture(52, 7, 1920, 1200));
    assert(!s.begin_publication(52, 7, 1920, 1200));
    auto result = s.snapshot();
    assert(result.phase == State::Phase::Done && result.model_computed && result.export_completed);
    assert(result.drawable_presented && result.original_rendering_recovered);
    assert(result.source_frame == 12 && result.publication_frame == 50);
  }
  for (int stage = 0; stage < 4; ++stage) {
    State s; assert(s.begin_capture(12, 7, 1920, 1200));
    if (stage > 0) assert(s.capture_completed(true));
    if (stage > 1) assert(s.compute_completed(true));
    if (stage > 2) assert(s.begin_publication(13, 7, 1920, 1200));
    s.fail();
    assert(!s.capture_completed(true)); assert(!s.compute_completed(true));
    assert(!s.begin_publication(14, 7, 1920, 1200)); assert(!s.publication_completed(true));
    assert(!s.note_presented(13, 7)); assert(s.snapshot().phase == State::Phase::Failed);
  }
  {
    State s; ready(s);
    assert(s.observe_geometry(8, 100, 100));
    assert(!s.observe_geometry(7, 1920, 1080));
    assert(!s.begin_publication(13, 7, 1920, 1200));
  }
  {
    State s; ready(s);
    std::atomic<unsigned> admitted{0}; std::vector<std::thread> threads;
    for (unsigned i = 0; i < 16; ++i) threads.emplace_back([&s, &admitted, i] {
      if (s.begin_publication(20 + i, 7, 1920, 1200)) ++admitted;
    });
    for (auto& t : threads) t.join();
    assert(admitted == 1);
    assert(s.commit_publication());
    assert(s.publication_completed(true));
    assert(!s.snapshot().drawable_presented); // Computed/exported is not shown.
    auto frame = s.snapshot().publication_frame;
    assert(s.original_completed(frame + 1, 7, true));
    assert(s.note_presented(frame, 7)); // Late callback cannot re-arm publication.
    assert(!s.begin_publication(frame + 2, 7, 1920, 1200));
  }
  {
    State s;ready(s);assert(s.begin_publication(13,7,1920,1200));s.fail();
    assert(!s.commit_publication());assert(!s.snapshot().publication_may_be_encoded);
  }
  {
    State s;assert(s.begin_capture(12,7,1920,1200));assert(s.capture_completed(true));
    assert(s.note_model_computed());s.fail();
    assert(s.snapshot().model_computed && !s.snapshot().export_completed && !s.snapshot().drawable_presented);
  }
  {
    State s;ready(s);assert(s.begin_publication(13,7,1920,1200));assert(s.commit_publication());
    s.fail();assert(s.snapshot().publication_may_be_encoded);
    assert(!s.publication_completed(true));assert(s.snapshot().phase==State::Phase::Failed);
    assert(s.note_presented(13,7));assert(s.original_completed(14,7,true));
    assert(s.snapshot().phase==State::Phase::Failed && s.snapshot().original_rendering_recovered);
  }
  std::puts("PASS delayed one-shot CPU admission, cancellation, identity, recovery and concurrent publication tests; no GPU/model/presentation claim");
}
