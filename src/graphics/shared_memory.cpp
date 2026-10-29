/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <utility>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/assert.h>
#include <rex/bit.h>
#include <rex/dbg.h>
#include <rex/graphics/shared_memory.h>
#include <rex/math.h>
#include <rex/memory.h>

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long long __stdcall GetTickCount64();
#endif

namespace {
// VR fork: the hot pages' clock, read for every range requested (several a draw). Its 16 ms steps are enough for
// heat that decays over a second; QueryPerformanceCounter there cost ~2% of the GPU thread.
uint32_t HotPageClockMs() {
#ifdef _WIN32
  return uint32_t(GetTickCount64());
#else
  return uint32_t(rex::chrono::Clock::QueryHostUptimeMillis());
#endif
}
}  // namespace

REXCVAR_DEFINE_UINT32(shared_memory_request_granularity_log2, 0, "GPU",
                      "log2 of the chunk to which requests for guest memory are widened before "
                      "uploading and watching it; 0 (or anything not above the page size) "
                      "requests exactly what a draw needs. Larger chunks mean fewer protection "
                      "changes and more data uploaded per change");

REXCVAR_DEFINE_UINT32(shared_memory_invalidation_pages_log2, 6, "GPU",
                      "A guest write to watched memory invalidates up to 2^n pages around it "
                      "(6 = the whole block of 64, the original behaviour)");
REXCVAR_DEFINE_UINT32(shared_memory_hot_page_faults, 0, "GPU",
                      "Guest write faults on one page, within about a second, after which the "
                      "page is treated as hot: not watched, and uploaded whenever it is "
                      "requested. 0 never does. For hosts where a fault and the protection "
                      "changes around it are expensive");
REXCVAR_DEFINE_UINT32(shared_memory_hot_page_exact, 1, "GPU",
                      "1: of a hot page, upload only the bytes a draw asks for (where the "
                      "graphics backend can); 0: the whole page");
REXCVAR_DEFINE_UINT32(shared_memory_hot_page_ms, 2000, "GPU",
                      "How long a hot page stays hot before it is watched again");
REXCVAR_DEFINE_BOOL(shared_memory_hot_page_compare, true, "GPU",
                    "Of hot pages uploaded whole (shared_memory_hot_page_exact 0), upload only the ones that changed "
                    "since their last upload (a copy of each is kept to compare)");

namespace rex::graphics {

// Guest memory copied to the GPU buffer so far, for performance diagnosis.
std::atomic<uint64_t> g_shared_memory_uploaded_bytes{0};
uint64_t SharedMemoryUploadedBytes() {
  return g_shared_memory_uploaded_bytes.load(std::memory_order_relaxed);
}
// Of that, the part that is hot pages; the bytes the draws actually asked for
// within those pages; and the number of requests.
std::atomic<uint64_t> g_shared_memory_hot_uploaded_bytes{0};
std::atomic<uint64_t> g_shared_memory_hot_asked_bytes{0};
std::atomic<uint64_t> g_shared_memory_requests{0};
// Why tracked (not hot) pages were uploaded, in bytes, by PageState; bytes
// invalidated by a write fault and by an explicit call; pages turned hot.
enum PageState : uint8_t {
  kPageNeverUploaded,
  kPageInvalidatedByFault,
  kPageInvalidatedExplicitly,
  kPageUploaded,
  kPageStateCount,
};
std::atomic<uint64_t> g_shared_memory_uploaded_by_state[kPageStateCount];
std::atomic<uint64_t> g_shared_memory_fault_invalidated_bytes{0};
std::atomic<uint64_t> g_shared_memory_explicitly_invalidated_bytes{0};
std::atomic<uint64_t> g_shared_memory_pages_turned_hot{0};
void SharedMemoryCounters(uint64_t out[10]) {
  out[0] = g_shared_memory_hot_uploaded_bytes.load(std::memory_order_relaxed);
  out[1] = g_shared_memory_hot_asked_bytes.load(std::memory_order_relaxed);
  out[2] = g_shared_memory_requests.load(std::memory_order_relaxed);
  for (int i = 0; i < kPageStateCount; ++i) {
    out[3 + i] = g_shared_memory_uploaded_by_state[i].load(std::memory_order_relaxed);
  }
  out[7] = g_shared_memory_fault_invalidated_bytes.load(std::memory_order_relaxed);
  out[8] = g_shared_memory_explicitly_invalidated_bytes.load(std::memory_order_relaxed);
  out[9] = g_shared_memory_pages_turned_hot.load(std::memory_order_relaxed);
}

SharedMemory::SharedMemory(memory::Memory& memory) : memory_(memory) {
  page_size_log2_ = rex::log2_ceil(uint32_t(rex::memory::page_size()));
}

SharedMemory::~SharedMemory() {
  ShutdownCommon();
}

void SharedMemory::InitializeCommon() {
  num_system_page_flags_ = ((kBufferSize >> page_size_log2_) + 63) / 64;
  system_page_flags_valid_.assign(num_system_page_flags_, 0);
  system_page_flags_valid_and_gpu_written_.assign(num_system_page_flags_, 0);
  page_heat_.assign(size_t(num_system_page_flags_) * 64, 0);
  page_state_.assign(size_t(num_system_page_flags_) * 64, kPageNeverUploaded);
  page_hot_until_ms_.assign(size_t(num_system_page_flags_) * 64, 0);

  memory_invalidation_callback_handle_ =
      memory_.RegisterPhysicalMemoryInvalidationCallback(MemoryInvalidationCallbackThunk, this);
}

void SharedMemory::InitializeSparseHostGpuMemory(uint32_t granularity_log2) {
  assert_true(granularity_log2 <= kBufferSizeLog2);
  assert_true(host_gpu_memory_sparse_granularity_log2_ == UINT32_MAX);
  host_gpu_memory_sparse_granularity_log2_ = granularity_log2;
  host_gpu_memory_sparse_allocated_.resize(
      size_t(1) << (std::max(kBufferSizeLog2 - granularity_log2, uint32_t(6)) - 6));
}

void SharedMemory::ShutdownCommon() {
  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, false);
  assert_true(global_watches_.empty());
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();

  if (memory_invalidation_callback_handle_ != nullptr) {
    memory_.UnregisterPhysicalMemoryInvalidationCallback(memory_invalidation_callback_handle_);
    memory_invalidation_callback_handle_ = nullptr;
  }

  if (host_gpu_memory_sparse_used_bytes_) {
    host_gpu_memory_sparse_used_bytes_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_used_mb", 0);
  }
  if (host_gpu_memory_sparse_allocations_) {
    host_gpu_memory_sparse_allocations_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_allocations", 0);
  }
  host_gpu_memory_sparse_allocated_.clear();
  host_gpu_memory_sparse_allocated_.shrink_to_fit();
  host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;

  system_page_flags_valid_.clear();
  system_page_flags_valid_.shrink_to_fit();
  system_page_flags_valid_and_gpu_written_.clear();
  system_page_flags_valid_and_gpu_written_.shrink_to_fit();
  num_system_page_flags_ = 0;
}

void SharedMemory::InvalidateAllPages() {
  auto global_lock = global_critical_region_.Acquire();

  std::fill(system_page_flags_valid_.begin(), system_page_flags_valid_.end(), uint64_t(0));
  std::fill(system_page_flags_valid_and_gpu_written_.begin(),
            system_page_flags_valid_and_gpu_written_.end(), uint64_t(0));
}

void SharedMemory::SetSystemPageBlocksValidWithGpuDataWritten() {
  auto global_lock = global_critical_region_.Acquire();

  // Pages that are valid only because the CPU uploaded them lose their valid
  // bit here, so the next frame re-reads them from guest memory.
  system_page_flags_valid_ = system_page_flags_valid_and_gpu_written_;
}

void SharedMemory::ClearCache() {
  // Keeping GPU-written data, so "invalidated by GPU".
  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, true);
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();
  SetSystemPageBlocksValidWithGpuDataWritten();
}

SharedMemory::GlobalWatchHandle SharedMemory::RegisterGlobalWatch(GlobalWatchCallback callback,
                                                                  void* callback_context) {
  GlobalWatch* watch = new GlobalWatch;
  watch->callback = callback;
  watch->callback_context = callback_context;

  auto global_lock = global_critical_region_.Acquire();
  global_watches_.push_back(watch);

  return reinterpret_cast<GlobalWatchHandle>(watch);
}

void SharedMemory::UnregisterGlobalWatch(GlobalWatchHandle handle) {
  auto watch = reinterpret_cast<GlobalWatch*>(handle);

  {
    auto global_lock = global_critical_region_.Acquire();
    auto it = std::find(global_watches_.begin(), global_watches_.end(), watch);
    assert_false(it == global_watches_.end());
    if (it != global_watches_.end()) {
      global_watches_.erase(it);
    }
  }

  delete watch;
}

SharedMemory::WatchHandle SharedMemory::WatchMemoryRange(uint32_t start, uint32_t length,
                                                         WatchCallback callback,
                                                         void* callback_context,
                                                         void* callback_data,
                                                         uint64_t callback_argument) {
  if (length == 0 || start >= kBufferSize) {
    return nullptr;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t watch_page_first = start >> page_size_log2_;
  uint32_t watch_page_last = (start + length - 1) >> page_size_log2_;
  uint32_t bucket_first = watch_page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  uint32_t bucket_last = watch_page_last << page_size_log2_ >> kWatchBucketSizeLog2;

  auto global_lock = global_critical_region_.Acquire();

  // Allocate the range.
  WatchRange* range = watch_range_first_free_;
  if (range != nullptr) {
    watch_range_first_free_ = range->next_free;
  } else {
    if (watch_range_pools_.empty() || watch_range_current_pool_allocated_ >= kWatchRangePoolSize) {
      watch_range_pools_.push_back(new WatchRange[kWatchRangePoolSize]);
      watch_range_current_pool_allocated_ = 0;
    }
    range = &(watch_range_pools_.back()[watch_range_current_pool_allocated_++]);
  }
  range->callback = callback;
  range->callback_context = callback_context;
  range->callback_data = callback_data;
  range->callback_argument = callback_argument;
  range->page_first = watch_page_first;
  range->page_last = watch_page_last;

  // Allocate and link the nodes.
  WatchNode* node_previous = nullptr;
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_node_first_free_;
    if (node != nullptr) {
      watch_node_first_free_ = node->next_free;
    } else {
      if (watch_node_pools_.empty() || watch_node_current_pool_allocated_ >= kWatchNodePoolSize) {
        watch_node_pools_.push_back(new WatchNode[kWatchNodePoolSize]);
        watch_node_current_pool_allocated_ = 0;
      }
      node = &(watch_node_pools_.back()[watch_node_current_pool_allocated_++]);
    }
    node->range = range;
    node->range_node_next = nullptr;
    if (node_previous != nullptr) {
      node_previous->range_node_next = node;
    } else {
      range->node_first = node;
    }
    node_previous = node;
    node->bucket_node_previous = nullptr;
    node->bucket_node_next = watch_buckets_[i];
    if (watch_buckets_[i] != nullptr) {
      watch_buckets_[i]->bucket_node_previous = node;
    }
    watch_buckets_[i] = node;
  }

  return reinterpret_cast<WatchHandle>(range);
}

void SharedMemory::UnwatchMemoryRange(WatchHandle handle) {
  auto global_lock = global_critical_region_.Acquire();
  UnlinkWatchRange(reinterpret_cast<WatchRange*>(handle));
}

void SharedMemory::FireWatches(uint32_t page_first, uint32_t page_last, bool invalidated_by_gpu) {
  uint32_t address_first = page_first << page_size_log2_;
  uint32_t address_last = (page_last << page_size_log2_) + ((1 << page_size_log2_) - 1);
  uint32_t bucket_first = address_first >> kWatchBucketSizeLog2;
  uint32_t bucket_last = address_last >> kWatchBucketSizeLog2;

  auto global_lock = global_critical_region_.Acquire();

  // Fire global watches.
  for (const auto global_watch : global_watches_) {
    global_watch->callback(global_lock, global_watch->callback_context, address_first, address_last,
                           invalidated_by_gpu);
  }

  // Fire per-range watches.
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_buckets_[i];
    while (node != nullptr) {
      WatchRange* range = node->range;
      // Store the next node now since when the callback is triggered, the links
      // will be broken.
      node = node->bucket_node_next;
      if (page_first <= range->page_last && page_last >= range->page_first) {
        range->callback(global_lock, range->callback_context, range->callback_data,
                        range->callback_argument, invalidated_by_gpu);
        UnlinkWatchRange(range);
      }
    }
  }
}

void SharedMemory::VrMarkEyeDependent(uint32_t start, uint32_t length, bool eye_dependent) {
  if (!length || start >= kBufferSize) {
    return;
  }
  if (vr_eye_dependent_pages_.empty()) {
    if (!eye_dependent) {
      return;
    }
    vr_eye_dependent_pages_.assign(((kBufferSize >> page_size_log2_) + 63) / 64, 0);
  }
  length = std::min(length, kBufferSize - start);
  const uint32_t first = start >> page_size_log2_, last = (start + length - 1) >> page_size_log2_;
  for (uint32_t page = first; page <= last; ++page) {
    const uint64_t bit = uint64_t(1) << (page & 63);
    vr_eye_dependent_pages_[page >> 6] = eye_dependent ? (vr_eye_dependent_pages_[page >> 6] | bit)
                                                       : (vr_eye_dependent_pages_[page >> 6] & ~bit);
  }
}

bool SharedMemory::VrIsRangeEyeDependent(uint32_t start, uint32_t length) const {
  if (vr_eye_dependent_pages_.empty() || !length || start >= kBufferSize) {
    return false;
  }
  length = std::min(length, kBufferSize - start);
  const uint32_t first = start >> page_size_log2_, last = (start + length - 1) >> page_size_log2_;
  for (uint32_t page = first; page <= last; ++page) {
    if (vr_eye_dependent_pages_[page >> 6] & (uint64_t(1) << (page & 63))) {
      return true;
    }
  }
  return false;
}

bool SharedMemory::IsRangeGpuWritten(uint32_t start, uint32_t length) const {
  if (!length || start >= kBufferSize) {
    return false;
  }
  length = std::min(length, kBufferSize - start);
  const uint32_t first = start >> page_size_log2_, last = (start + length - 1) >> page_size_log2_;
  for (uint32_t page = first; page <= last; ++page) {
    if (system_page_flags_valid_and_gpu_written_[page >> 6] & (uint64_t(1) << (page & 63))) {
      return true;
    }
  }
  return false;
}

void SharedMemory::RangeWrittenByGpu(uint32_t start, uint32_t length) {
  if (length == 0 || start >= kBufferSize) {
    return;
  }
  TrackSubmission();  // VR fork: written by the submission
  TrackPages(vr_touched_pages_, start, length);
  if (!hot_page_copies_.empty()) {  // VR fork: the GPU's data, not the guest's: no hot page copy holds it
    for (uint32_t page = start >> page_size_log2_; page <= (start + length - 1) >> page_size_log2_; ++page) {
      hot_page_copies_.erase(page);
    }
  }
  length = std::min(length, kBufferSize - start);
  uint32_t end = start + length - 1;
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = end >> page_size_log2_;

  // Trigger modification callbacks so, for instance, resolved data is loaded to
  // the texture.
  FireWatches(page_first, page_last, true);

  // Mark the range as valid (so pages are not reuploaded until modified by the
  // CPU) and watch it so the CPU can reuse it and this will be caught.
  MakeRangeValid(start, length, true);
}

bool SharedMemory::AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                    uint32_t length_allocations) {
  assert_always(
      "Sparse host GPU memory allocation has been initialized, but the "
      "implementation doesn't provide AllocateSparseHostGpuMemoryRange");
  return false;
}

void SharedMemory::MakeRangeValid(uint32_t start, uint32_t length, bool written_by_gpu) {
  if (length == 0 || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t last = start + length - 1;
  uint32_t valid_page_first = start >> page_size_log2_;
  uint32_t valid_page_last = last >> page_size_log2_;
  uint32_t valid_block_first = valid_page_first >> 6;
  uint32_t valid_block_last = valid_page_last >> 6;

  // Data uploaded from guest memory to a hot page is good for this request
  // only: the page is not marked valid and not watched (data the GPU wrote is
  // different, it exists nowhere else, so it is tracked as usual).
  const uint32_t now_ms = HotPageClockMs();
  bool skipped_hot_pages = false;
  {
    auto global_lock = global_critical_region_.Acquire();
    const bool check_hot = !written_by_gpu && AnyPageHot(now_ms);

    for (uint32_t i = valid_block_first; i <= valid_block_last; ++i) {
      uint64_t valid_bits = UINT64_MAX;
      if (i == valid_block_first) {
        valid_bits &= ~((uint64_t(1) << (valid_page_first & 63)) - 1);
      }
      if (i == valid_block_last && (valid_page_last & 63) != 63) {
        valid_bits &= (uint64_t(1) << ((valid_page_last & 63) + 1)) - 1;
      }
      if (check_hot) {
        const uint64_t hot_bits = HotPageMask(i, now_ms) & valid_bits;
        if (hot_bits) {
          skipped_hot_pages = true;
          valid_bits &= ~hot_bits;
        }
      }
      system_page_flags_valid_[i] |= valid_bits;
      uint64_t& gpu_written = system_page_flags_valid_and_gpu_written_[i];
      gpu_written = written_by_gpu ? (gpu_written | valid_bits) : (gpu_written & ~valid_bits);
      if (!written_by_gpu && !vr_eye_dependent_pages_.empty()) {  // VR fork: CPU data, the same in both eyes
        vr_eye_dependent_pages_[i] &= ~valid_bits;
      }
    }
  }

  if (memory_invalidation_callback_handle_) {
    if (!skipped_hot_pages) {
      memory().EnablePhysicalMemoryAccessCallbacks(
          valid_page_first << page_size_log2_,
          (valid_page_last - valid_page_first + 1) << page_size_log2_, true, false);
    } else {
      // Watch the runs of pages between the hot ones.
      uint32_t run_first = UINT32_MAX;
      for (uint32_t page = valid_page_first; page <= valid_page_last + 1; ++page) {
        const bool watch = page <= valid_page_last && !IsPageHot(page, now_ms);
        if (watch) {
          if (run_first == UINT32_MAX) run_first = page;
        } else if (run_first != UINT32_MAX) {
          memory().EnablePhysicalMemoryAccessCallbacks(run_first << page_size_log2_,
                                                       (page - run_first) << page_size_log2_, true,
                                                       false);
          run_first = UINT32_MAX;
        }
      }
    }
  }
}

uint64_t SharedMemory::HotPageMask(uint32_t block, uint32_t now_ms) const {
  uint64_t mask = 0;
  const uint32_t* until = &page_hot_until_ms_[size_t(block) * 64];
  for (uint32_t n = 0; n < 64; ++n) {
    if (int32_t(until[n] - now_ms) > 0) {
      mask |= uint64_t(1) << n;
    }
  }
  return mask;
}

void SharedMemory::UnlinkWatchRange(WatchRange* range) {
  uint32_t bucket = range->page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  WatchNode* node = range->node_first;
  while (node != nullptr) {
    WatchNode* node_next = node->range_node_next;
    if (node->bucket_node_previous != nullptr) {
      node->bucket_node_previous->bucket_node_next = node->bucket_node_next;
    } else {
      watch_buckets_[bucket] = node->bucket_node_next;
    }
    if (node->bucket_node_next != nullptr) {
      node->bucket_node_next->bucket_node_previous = node->bucket_node_previous;
    }
    node->next_free = watch_node_first_free_;
    watch_node_first_free_ = node;
    node = node_next;
    ++bucket;
  }
  range->next_free = watch_range_first_free_;
  watch_range_first_free_ = range;
}

bool SharedMemory::RequestRanges(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  const bool result = RequestRangesUntracked(ranges, count);
  if (ranges) {  // VR fork: read by the submission (after its uploads)
    TrackSubmission();
    for (size_t i = 0; i < count; ++i) {
      TrackPages(vr_touched_pages_, ranges[i].first, ranges[i].second);
    }
  }
  return result;
}

void SharedMemory::TrackSubmission() {
  const uint64_t submission = TrackingSubmission();
  if (vr_touched_pages_.empty()) {
    const size_t words = ((kBufferSize >> page_size_log2_) + 63) / 64;
    vr_touched_pages_.assign(words, 0);
    vr_hoisted_pages_.assign(words, 0);
    vr_tracked_submission_ = submission;
  } else if (submission != vr_tracked_submission_) {
    std::fill(vr_touched_pages_.begin(), vr_touched_pages_.end(), 0);
    std::fill(vr_hoisted_pages_.begin(), vr_hoisted_pages_.end(), 0);
    vr_tracked_submission_ = submission;
  }
}

void SharedMemory::TrackPages(std::vector<uint64_t>& bitmap, uint32_t start, uint32_t length) {
  if (!length || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  const uint32_t first = start >> page_size_log2_, last = (start + length - 1) >> page_size_log2_;
  for (uint32_t page = first; page <= last; ++page) {
    bitmap[page >> 6] |= uint64_t(1) << (page & 63);
  }
}

bool SharedMemory::HoistUpload(uint32_t page_first, uint32_t page_count) {
  TrackSubmission();
  for (uint32_t page = page_first; page < page_first + page_count; ++page) {
    const uint64_t bit = uint64_t(1) << (page & 63);
    if ((vr_touched_pages_[page >> 6] | vr_hoisted_pages_[page >> 6]) & bit) {
      return false;
    }
  }
  for (uint32_t page = page_first; page < page_first + page_count; ++page) {
    vr_hoisted_pages_[page >> 6] |= uint64_t(1) << (page & 63);
  }
  return true;
}

bool SharedMemory::RequestRangesUntracked(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  if (ranges == nullptr || !count) {
    return true;
  }

  // Some texture or buffer is empty, for example - safe to draw in this case.
  std::vector<std::pair<uint32_t, uint32_t>>& merged_ranges = request_merged_ranges_;
  merged_ranges.clear();
  // A write to watched memory invalidates a wide area at once (the block of 64
  // pages around it, see MemoryInvalidationCallback), but each request then
  // uploads and re-watches only the pages it needs, so one fault is followed
  // by a protection change for every draw that comes back for its piece.
  // Where a protection change is expensive (26 us on PS5 whatever its size),
  // widening the request to whole chunks makes the re-watch as coarse as the
  // invalidation: the invalid pages of the chunk are uploaded and watched in
  // one go, and pages already valid are left alone as before.
  // The ranges as asked for, before widening: hot pages are uploaded only
  // where a draw really uses them.
  std::vector<std::pair<uint32_t, uint32_t>>& requested_ranges = request_asked_ranges_;
  requested_ranges.clear();
  const uint32_t request_granularity_log2 =
      std::min(uint32_t(REXCVAR_GET(shared_memory_request_granularity_log2)), uint32_t(24));
  const uint32_t request_chunk_mask =
      request_granularity_log2 > page_size_log2_ ? (uint32_t(1) << request_granularity_log2) - 1
                                                 : 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t start = ranges[i].first;
    uint32_t length = ranges[i].second;
    if (!length) {
      continue;
    }
    if (start > kBufferSize || (kBufferSize - start) < length) {
      return false;
    }
    requested_ranges.emplace_back(start, length);
    if (request_chunk_mask) {
      const uint64_t end = std::min(
          (uint64_t(start) + length + request_chunk_mask) & ~uint64_t(request_chunk_mask),
          uint64_t(kBufferSize));
      start &= ~request_chunk_mask;
      length = uint32_t(end - start);
    }
    merged_ranges.emplace_back(start, length);
  }
  if (merged_ranges.empty()) {
    return true;
  }

  SCOPE_profile_cpu_f("gpu");
  g_shared_memory_requests.fetch_add(1, std::memory_order_relaxed);

  std::sort(merged_ranges.begin(), merged_ranges.end(),
            [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
              return a.first < b.first;
            });
  size_t merged_write = 0;
  for (size_t i = 1; i < merged_ranges.size(); ++i) {
    std::pair<uint32_t, uint32_t>& range_previous = merged_ranges[merged_write];
    const std::pair<uint32_t, uint32_t>& range_current = merged_ranges[i];
    uint64_t previous_end = uint64_t(range_previous.first) + uint64_t(range_previous.second);
    uint64_t current_start = uint64_t(range_current.first);
    if (current_start <= previous_end) {
      uint64_t current_end = current_start + uint64_t(range_current.second);
      if (current_end > previous_end) {
        range_previous.second = uint32_t(current_end - uint64_t(range_previous.first));
      }
    } else {
      merged_ranges[++merged_write] = range_current;
    }
  }
  merged_ranges.resize(merged_write + 1);

  for (const std::pair<uint32_t, uint32_t>& range : merged_ranges) {
    if (!EnsureHostGpuMemoryAllocated(range.first, range.second)) {
      return false;
    }
  }

  upload_ranges_.clear();
  hot_byte_ranges_.clear();
  const bool hot_exact = REXCVAR_GET(shared_memory_hot_page_exact) != 0;
  auto append_upload_range = [this](uint32_t page_start, uint32_t page_count) {
    if (!page_count) {
      return;
    }
    for (uint32_t page = page_start; page < page_start + page_count; ++page) {
      g_shared_memory_uploaded_by_state[page_state_[page]].fetch_add(
          uint64_t(1) << page_size_log2_, std::memory_order_relaxed);
      page_state_[page] = kPageUploaded;
      if (!hot_page_copies_.empty()) {
        hot_page_copies_.erase(page);  // VR fork: uploaded from a watch, not as hot
      }
    }
    if (!upload_ranges_.empty()) {
      std::pair<uint32_t, uint32_t>& last_upload_range = upload_ranges_.back();
      if (last_upload_range.first + last_upload_range.second == page_start) {
        last_upload_range.second += page_count;
        return;
      }
    }
    upload_ranges_.emplace_back(page_start, page_count);
  };
  {
    auto global_lock = global_critical_region_.Acquire();
    const uint32_t now_ms = HotPageClockMs();
    const bool any_hot = AnyPageHot(now_ms);
    for (const std::pair<uint32_t, uint32_t>& range : merged_ranges) {
      uint32_t page_first = range.first >> page_size_log2_;
      uint32_t page_last = (range.first + range.second - 1) >> page_size_log2_;
      uint32_t block_first = page_first >> 6;
      uint32_t block_last = page_last >> 6;
      uint32_t range_start = UINT32_MAX;
      for (uint32_t i = block_first; i <= block_last; ++i) {
        uint64_t block_valid = system_page_flags_valid_[i];
        // Hot pages are handled below, from the ranges as asked for.
        if (any_hot) {
          block_valid |= HotPageMask(i, now_ms);
        }
        // Consider pages in the block outside the requested range valid.
        if (i == block_first) {
          uint64_t block_before = (uint64_t(1) << (page_first & 63)) - 1;
          block_valid |= block_before;
        }
        if (i == block_last && (page_last & 63) != 63) {
          uint64_t block_inside = (uint64_t(1) << ((page_last & 63) + 1)) - 1;
          block_valid |= ~block_inside;
        }

        while (true) {
          uint32_t block_page;
          if (range_start == UINT32_MAX) {
            // Check if need to open a new range.
            if (!rex::bit_scan_forward(~block_valid, &block_page)) {
              break;
            }
            range_start = (i << 6) + block_page;
          } else {
            // Check if need to close the range.
            // Ignore the valid pages before the beginning of the range.
            uint64_t block_valid_from_start = block_valid;
            if (i == (range_start >> 6)) {
              block_valid_from_start &= ~((uint64_t(1) << (range_start & 63)) - 1);
            }
            if (!rex::bit_scan_forward(block_valid_from_start, &block_page)) {
              break;
            }
            append_upload_range(range_start, (i << 6) + block_page - range_start);
            // In the next iteration within this block, consider this range
            // valid since it has been queued for upload.
            block_valid |= (uint64_t(1) << block_page) - 1;
            range_start = UINT32_MAX;
          }
        }
      }
      if (range_start != UINT32_MAX) {
        append_upload_range(range_start, page_last + 1 - range_start);
      }
    }
    // Hot pages: always uploaded where a draw uses them (never where the
    // request was only widened over them), unless the GPU wrote them, in which
    // case guest memory does not have the data. Anything watching such a page
    // (a texture loaded from it) is told it may have changed, since no fault
    // will say so.
    if (any_hot) {
      for (const std::pair<uint32_t, uint32_t>& range : requested_ranges) {
        const uint32_t page_first = range.first >> page_size_log2_;
        const uint32_t page_last = (range.first + range.second - 1) >> page_size_log2_;
        uint32_t run_first = UINT32_MAX;
        for (uint32_t page = page_first; page <= page_last + 1; ++page) {
          const bool upload =
              page <= page_last && IsPageHot(page, now_ms) &&
              !(system_page_flags_valid_and_gpu_written_[page >> 6] & (uint64_t(1) << (page & 63)));
          if (upload) {
            if (run_first == UINT32_MAX) run_first = page;
          } else if (run_first != UINT32_MAX) {
            const uint64_t run_start = uint64_t(run_first) << page_size_log2_;
            const uint64_t run_end = uint64_t(page) << page_size_log2_;
            const uint64_t asked_start = std::max(run_start, uint64_t(range.first));
            const uint64_t asked_end = std::min(run_end, uint64_t(range.first) + range.second);
            g_shared_memory_hot_asked_bytes.fetch_add(asked_end - asked_start,
                                                      std::memory_order_relaxed);
            if (hot_exact) {
              // Only what is used, to 64-byte boundaries (a page is a multiple).
              const uint32_t byte_start = uint32_t(asked_start) & ~uint32_t(63);
              const uint32_t byte_end = uint32_t((asked_end + 63) & ~uint64_t(63));
              hot_byte_ranges_.emplace_back(byte_start, byte_end - byte_start);
            } else if (REXCVAR_GET(shared_memory_hot_page_compare)) {
              // VR fork: only the pages that changed since their last upload.
              const uint32_t page_bytes = uint32_t(1) << page_size_log2_;
              uint32_t changed_first = UINT32_MAX;
              for (uint32_t p = run_first; p <= page; ++p) {
                bool changed = false;
                if (p < page) {
                  const uint8_t* guest = memory().TranslatePhysical(p << page_size_log2_);
                  auto& copy = hot_page_copies_[p];
                  // Only the bytes this draw asked for (a changed page is uploaded and kept whole, so the copy
                  // stays what the GPU has).
                  const uint32_t page_start = p << page_size_log2_;
                  const uint32_t asked_first = std::max(page_start, range.first) - page_start;
                  const uint32_t asked_end = std::min(page_start + page_bytes, range.first + range.second) - page_start;
                  changed = !copy || std::memcmp(copy.get() + asked_first, guest + asked_first,
                                                 asked_end - asked_first) != 0;
                  static const bool vr_count = std::getenv("MCLA_VR_COUNT") != nullptr;  // development
                  if (vr_count) {
                    static uint64_t pages = 0, changed_pages = 0, blocks = 0, new_pages = 0;
                    static auto window = std::chrono::steady_clock::now();
                    ++pages;
                    if (changed && copy) {
                      ++changed_pages;
                      for (uint32_t b = 0; b < page_bytes; b += 64) {
                        blocks += std::memcmp(copy.get() + b, guest + b, 64) != 0;
                      }
                    } else if (changed) {
                      ++new_pages;
                    }
                    if (std::chrono::steady_clock::now() - window > std::chrono::seconds(2)) {
                      REXGPU_INFO("hot pages compared: {}, changed {} (+{} new), 64-byte blocks changed {:.1f} a page",
                                  pages, changed_pages, new_pages, changed_pages ? double(blocks) / changed_pages : 0.0);
                      pages = changed_pages = blocks = new_pages = 0;
                      window = std::chrono::steady_clock::now();
                    }
                  }
                  if (changed) {
                    if (!copy) {
                      copy = std::make_unique<uint8_t[]>(page_bytes);
                    }
                    std::memcpy(copy.get(), guest, page_bytes);
                  }
                }
                if (changed && changed_first == UINT32_MAX) {
                  changed_first = p;
                } else if (!changed && changed_first != UINT32_MAX) {
                  upload_ranges_.emplace_back(changed_first, p - changed_first);
                  g_shared_memory_hot_uploaded_bytes.fetch_add(uint64_t(p - changed_first) << page_size_log2_,
                                                               std::memory_order_relaxed);
                  FireWatches(changed_first, p - 1, false);
                  changed_first = UINT32_MAX;
                }
              }
            } else {
              upload_ranges_.emplace_back(run_first, page - run_first);
              g_shared_memory_hot_uploaded_bytes.fetch_add(run_end - run_start,
                                                           std::memory_order_relaxed);
            }
            if (hot_exact || !REXCVAR_GET(shared_memory_hot_page_compare)) {
              FireWatches(run_first, page - 1, false);
            }
            run_first = UINT32_MAX;
          }
        }
      }
    }
  }

  COUNT_profile_set("gpu/shared_memory/request_ranges_count", uint32_t(count));
  COUNT_profile_set("gpu/shared_memory/request_ranges_merged_count",
                    uint32_t(merged_ranges.size()));
  COUNT_profile_set("gpu/shared_memory/request_ranges_upload_count",
                    uint32_t(upload_ranges_.size()));

  if (upload_ranges_.empty() && hot_byte_ranges_.empty()) {
    return true;
  }

  uint64_t upload_pages = 0;
  for (const std::pair<uint32_t, uint32_t>& upload_range : upload_ranges_) {
    upload_pages += upload_range.second;
  }
  g_shared_memory_uploaded_bytes.fetch_add(upload_pages << page_size_log2_,
                                           std::memory_order_relaxed);

  std::sort(upload_ranges_.begin(), upload_ranges_.end());  // VR fork: hot pages come after the rest
  if (!UploadRanges(upload_ranges_)) {
    return false;
  }
  if (!hot_byte_ranges_.empty()) {
    uint64_t hot_bytes = 0;
    for (const std::pair<uint32_t, uint32_t>& byte_range : hot_byte_ranges_) {
      hot_bytes += byte_range.second;
    }
    if (UploadByteRanges(hot_byte_ranges_)) {
      g_shared_memory_uploaded_bytes.fetch_add(hot_bytes, std::memory_order_relaxed);
      g_shared_memory_hot_uploaded_bytes.fetch_add(hot_bytes, std::memory_order_relaxed);
    } else {
      // No byte path in this backend: the whole pages, as before.
      upload_ranges_.clear();
      for (const std::pair<uint32_t, uint32_t>& byte_range : hot_byte_ranges_) {
        const uint32_t page_first = byte_range.first >> page_size_log2_;
        const uint32_t page_last = (byte_range.first + byte_range.second - 1) >> page_size_log2_;
        upload_ranges_.emplace_back(page_first, page_last - page_first + 1);
      }
      std::sort(upload_ranges_.begin(), upload_ranges_.end());
      return UploadRanges(upload_ranges_);
    }
  }
  return true;
}

bool SharedMemory::RequestRange(uint32_t start, uint32_t length) {
  std::pair<uint32_t, uint32_t> range(start, length);
  return RequestRanges(&range, 1);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallbackThunk(
    void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range) {
  return reinterpret_cast<SharedMemory*>(context_ptr)
      ->MemoryInvalidationCallback(physical_address_start, length, exact_range);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallback(
    uint32_t physical_address_start, uint32_t length, bool exact_range) {
  if (length == 0 || physical_address_start >= kBufferSize) {
    return std::make_pair(uint32_t(0), UINT32_MAX);
  }
  length = std::min(length, kBufferSize - physical_address_start);
  uint32_t physical_address_last = physical_address_start + (length - 1);

  uint32_t page_first = physical_address_start >> page_size_log2_;
  uint32_t page_last = physical_address_last >> page_size_log2_;
  uint32_t block_first = page_first >> 6;
  uint32_t block_last = page_last >> 6;

  auto global_lock = global_critical_region_.Acquire();
  {  // VR fork, development (MCLA_VR_COUNT): invalidations by kind, every 2 s
    static const bool vr_count = std::getenv("MCLA_VR_COUNT") != nullptr;
    if (vr_count) {
      static uint64_t exact_calls = 0, exact_pages = 0, fault_calls = 0, fault_pages = 0;
      static auto window = std::chrono::steady_clock::now();
      (exact_range ? exact_calls : fault_calls) += 1;
      (exact_range ? exact_pages : fault_pages) += page_last - page_first + 1;
      if (std::chrono::steady_clock::now() - window > std::chrono::seconds(2)) {
        REXGPU_INFO("invalidations in 2 s: {} exact ({} pages), {} write faults ({} pages)", exact_calls, exact_pages,
                    fault_calls, fault_pages);
        exact_calls = exact_pages = fault_calls = fault_pages = 0;
        window = std::chrono::steady_clock::now();
      }
    }
  }

  // A guest write fault (not an exact-range invalidation, which is the GPU
  // side or an explicit request): count it against the pages written, and
  // make a page that keeps faulting hot.
  const uint32_t hot_threshold = REXCVAR_GET(shared_memory_hot_page_faults);
  if (!exact_range && hot_threshold) {
    const uint32_t now_ms = HotPageClockMs();
    if (uint32_t(now_ms - page_heat_decay_ms_) >= 1000) {
      page_heat_decay_ms_ = now_ms;
      for (uint8_t& heat : page_heat_) {
        heat >>= 1;
      }
    }
    const uint32_t hot_until = now_ms + std::max(REXCVAR_GET(shared_memory_hot_page_ms), uint32_t(1));
    for (uint32_t page = page_first; page <= page_last; ++page) {
      if (page_heat_[page] < 255) {
        ++page_heat_[page];
      }
      if (page_heat_[page] >= hot_threshold) {
        g_shared_memory_pages_turned_hot.fetch_add(1, std::memory_order_relaxed);
        page_heat_[page] = 0;
        page_hot_until_ms_[page] = hot_until;
        page_hot_latest_until_ms_ = hot_until;
      }
    }
  }

  // How far a write fault's invalidation may be widened: to the block of 64
  // pages by default, which is 256 KiB with 4 KiB pages but 1 MiB with 16 KiB
  // ones, all of it to be copied again.
  const uint32_t widen_mask =
      (uint32_t(1) << std::min(REXCVAR_GET(shared_memory_invalidation_pages_log2), uint32_t(6))) - 1;
  const uint32_t widen_first = page_first & ~widen_mask;
  const uint32_t widen_last = page_last | widen_mask;

  if (!exact_range) {
    // Check if a somewhat wider range (up to 256 KB with 4 KB pages) can be
    // invalidated - if no GPU-written data nearby that was not intended to be
    // invalidated since it's not in sync with CPU memory and can't be
    // reuploaded. It's a lot cheaper to upload some excess data than to catch
    // access violations - with 4 KB callbacks, 58410824 (being a
    // software-rendered game) runs at 4 FPS on Intel Core i7-3770, with 64 KB,
    // the CPU game code takes 3 ms to run per frame, but with 256 KB, it's
    // 0.7 ms.
    if (page_first & 63) {
      uint64_t gpu_written_start = system_page_flags_valid_and_gpu_written_[block_first];
      gpu_written_start &= (uint64_t(1) << (page_first & 63)) - 1;
      page_first = (page_first & ~uint32_t(63)) + (64 - rex::lzcnt(gpu_written_start));
    }
    if ((page_last & 63) != 63) {
      uint64_t gpu_written_end = system_page_flags_valid_and_gpu_written_[block_last];
      gpu_written_end &= ~((uint64_t(1) << ((page_last & 63) + 1)) - 1);
      page_last =
          (page_last & ~uint32_t(63)) + (std::max(rex::tzcnt(gpu_written_end), uint8_t(1)) - 1);
    }
    page_first = std::max(page_first, widen_first);
    page_last = std::min(page_last, widen_last);
  }

  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t invalidate_bits = UINT64_MAX;
    if (i == block_first) {
      invalidate_bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      invalidate_bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    system_page_flags_valid_[i] &= ~invalidate_bits;
    system_page_flags_valid_and_gpu_written_[i] &= ~invalidate_bits;
  }

  for (uint32_t page = page_first; page <= page_last; ++page) {
    page_state_[page] = exact_range ? kPageInvalidatedExplicitly : kPageInvalidatedByFault;
  }
  (exact_range ? g_shared_memory_explicitly_invalidated_bytes
               : g_shared_memory_fault_invalidated_bytes)
      .fetch_add(uint64_t(page_last - page_first + 1) << page_size_log2_,
                 std::memory_order_relaxed);

  FireWatches(page_first, page_last, false);

  return std::make_pair(page_first << page_size_log2_, (page_last - page_first + 1)
                                                           << page_size_log2_);
}

bool SharedMemory::EnsureHostGpuMemoryAllocated(uint32_t start, uint32_t length) {
  if (host_gpu_memory_sparse_granularity_log2_ == UINT32_MAX) {
    return true;
  }
  if (!length) {
    return true;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return false;
  }
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = (start + length - 1) >> page_size_log2_;
  uint32_t allocation_first =
      page_first << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_;
  uint32_t allocation_last =
      page_last << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_;
  while (true) {
    std::pair<size_t, size_t> allocation_range =
        rex::bit::GetNextRangeUnset(host_gpu_memory_sparse_allocated_.data(), allocation_first,
                                    allocation_last - allocation_first + 1);
    if (!allocation_range.second) {
      break;
    }
    if (!AllocateSparseHostGpuMemoryRange(uint32_t(allocation_range.first),
                                          uint32_t(allocation_range.second))) {
      return false;
    }
    rex::bit::SetRange(host_gpu_memory_sparse_allocated_.data(), allocation_range.first,
                       allocation_range.second);
    ++host_gpu_memory_sparse_allocations_;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_allocations",
                      host_gpu_memory_sparse_allocations_);
    host_gpu_memory_sparse_used_bytes_ += uint32_t(allocation_range.second)
                                          << host_gpu_memory_sparse_granularity_log2_;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_used_mb",
                      (host_gpu_memory_sparse_used_bytes_ + ((1 << 20) - 1)) >> 20);
    allocation_first = uint32_t(allocation_range.first + allocation_range.second);
  }
  return true;
}

}  // namespace rex::graphics
