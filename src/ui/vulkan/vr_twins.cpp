// VR fork: the right eye's twin resources and its replay function table (see vr_twins.h).

#include <rex/ui/vulkan/vr_twins.h>

#include <algorithm>
#include <intrin.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/logging.h>
#include <rex/ui/vulkan/vr_openxr.h>

namespace rex::ui::vulkan::vr {

namespace {

// A map from a Vulkan handle (never null) to a value: open addressing, linear probing, deletion by backward shift.
// The lookups run for every descriptor write and every replayed command; std::unordered_map's division and node hops
// were a visible share of the GPU thread. Elements move: nothing may keep a pointer to one.
template <typename K, typename V>
class FlatMap {
 public:
  using key_type = K;
  using mapped_type = V;
  struct Slot {
    K first{};
    V second{};
  };
  FlatMap() { Rehash(64); }
  const Slot* find(K key) const {
    if (!key) {
      return nullptr;
    }
    for (size_t i = Home(key);; i = (i + 1) & mask_) {
      const Slot& slot = slots_[i];
      if (slot.first == key) {
        return &slot;
      }
      if (!slot.first) {
        return nullptr;
      }
    }
  }
  Slot* find(K key) { return const_cast<Slot*>(static_cast<const FlatMap*>(this)->find(key)); }
  std::nullptr_t end() const { return nullptr; }
  size_t count(K key) const { return find(key) ? 1 : 0; }
  size_t size() const { return size_; }
  V& operator[](K key) {
    if ((size_ + 1) * 2 > slots_.size()) {
      Rehash(slots_.size() * 2);
    }
    size_t i = Home(key);
    while (slots_[i].first && slots_[i].first != key) {
      i = (i + 1) & mask_;
    }
    if (!slots_[i].first) {
      slots_[i].first = key;
      slots_[i].second = V{};
      ++size_;
    }
    return slots_[i].second;
  }
  void erase(Slot* slot) {
    size_t hole = size_t(slot - slots_.data());
    for (size_t j = (hole + 1) & mask_; slots_[j].first; j = (j + 1) & mask_) {
      const size_t home = Home(slots_[j].first);
      // The element at j may fill the hole unless its home lies cyclically in (hole, j].
      const bool stays = hole <= j ? (home > hole && home <= j) : (home > hole || home <= j);
      if (!stays) {
        slots_[hole] = slots_[j];
        hole = j;
      }
    }
    slots_[hole] = Slot{};
    --size_;
  }
  size_t erase(K key) {
    Slot* slot = find(key);
    if (!slot) {
      return 0;
    }
    erase(slot);
    return 1;
  }

 private:
  size_t Home(K key) const {
    return size_t((uint64_t(reinterpret_cast<uintptr_t>(key)) * 0x9E3779B97F4A7C15ull) >> shift_);
  }
  void Rehash(size_t capacity) {
    std::vector<Slot> old;
    old.swap(slots_);
    slots_.assign(capacity, Slot{});
    mask_ = capacity - 1;
    shift_ = 64;
    for (size_t c = capacity; c > 1; c >>= 1) {
      --shift_;
    }
    size_ = 0;
    for (const Slot& slot : old) {
      if (slot.first) {
        (*this)[slot.first] = slot.second;
      }
    }
  }
  std::vector<Slot> slots_;
  size_t mask_ = 0;
  uint32_t shift_ = 64;
  size_t size_ = 0;
};

VulkanDevice::Functions real;   // the device's own functions
VulkanDevice::Functions right;  // the right eye's replay table
uint32_t host_visible_types = 0;

// The resources' maps. The GPU thread is their only writer (writing under the lock, for the right eye's replay on
// the submission thread) and reads them without it. Other threads' resources (the presenter's, the UI's) are never
// replayed and get no twins; what they destroy of the GPU thread's is forgotten on the GPU thread (DrainForeign).
std::shared_mutex mutex;
thread_local bool gpu_thread = false;
std::mutex foreign_mutex;
std::vector<std::function<void()>> foreign_work;
std::atomic<bool> foreign_pending{false};
FlatMap<VkDeviceMemory, VkDeviceMemory> memory_twins;
thread_local bool sparse_twin_next = false;  // the next sparse buffer this thread creates gets a sparse twin
FlatMap<VkImage, VkImage> pending_images, image_twins;  // pending: made, not yet bound
FlatMap<VkBuffer, VkBuffer> pending_buffers, buffer_twins;
FlatMap<VkImageView, VkImageView> image_view_twins;
// Textures (MarkTexture) hold the same picture in both eyes until one is loaded from what the GPU wrote
// (MarkEyeDependent): their descriptors keep the left eye's view. Records of the images and views with twins (the
// views point at their image's record: elements of an unordered_map stay put).
struct ImageRecord {
  bool eye_independent = false;
};
std::unordered_map<VkImage, ImageRecord> image_records;
struct ViewRecord {
  VkImageView twin;
  const ImageRecord* image;
};
FlatMap<VkImageView, ViewRecord> view_records;
FlatMap<VkBufferView, VkBufferView> buffer_view_twins;
FlatMap<VkFramebuffer, VkFramebuffer> framebuffer_twins;
// Descriptor sets get a twin only once a write puts an eye-dependent resource in them (most of the per-draw texture
// sets never do); the right eye's replay binds the set itself otherwise. The same lock as the resources: a descriptor
// write takes it once.
std::shared_mutex& set_mutex = mutex;
FlatMap<VkDescriptorPool, VkDescriptorPool> pool_twins;
std::unordered_map<VkDescriptorPool, std::vector<VkDescriptorSet>> pool_sets;  // the sets with twins
FlatMap<VkDescriptorSet, VkDescriptorSet> set_twins;
// Twins kept while their set holds only what is the same in both eyes (the GPU thread recycles sets without
// resetting their pools; a set rewritten with eye-dependent content wakes its twin again).
FlatMap<VkDescriptorSet, VkDescriptorSet> dormant_twins;
// The sets this thread made lately (a set is written right after it is made): the layout a twin needs.
struct RecentSet {
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool;
  VkDescriptorSetLayout layout;
};
thread_local RecentSet recent_sets[64];
thread_local uint32_t recent_next = 0;
// Sets from pools never reset (made once, written whenever): their layouts kept until the pool is reset.
std::unordered_set<VkDescriptorPool> reset_pools;
std::unordered_map<VkDescriptorSet, RecentSet> lasting_sets;
std::unordered_map<VkDescriptorPool, std::vector<VkDescriptorSet>> lasting_pool_sets;
std::atomic<uint64_t> sets_made{0};
FlatMap<VkQueryPool, VkQueryPool> query_pool_twins;
std::atomic<uint64_t> twin_bytes{0};

thread_local int replay_eye = 0;

// Multiview: the layered images (a render target's two eyes), their left-eye views' views of both layers, the
// descriptor sets bound in their place in passes drawing both eyes. Under the same lock as the twins.
thread_local bool layered_next = false;
FlatMap<VkImage, uint32_t> layered_images;
FlatMap<VkImageView, VkImageView> array_views;
FlatMap<VkDescriptorSet, VkDescriptorSet> multiview_variants;
thread_local bool multiview_pass = false, last_pass_multiview = false;
// Multiview: the right eye's replay inside a pass of its own (not drawing both eyes). Outside one, the graphics state
// it would set is the left eye's table's to set (both eyes' passes draw with it): it leaves it out.
thread_local bool right_in_pass = false;
bool RightSkipsGraphics() { return replay_eye && !right_in_pass && Multiview(); }
VulkanDevice::Functions left;  // multiview: the left eye's (and the once-run passes') replay table

// Development (MCLA_VR_COUNT=1): the right eye's barriers and copies (one eye's worth), logged every 2 s.
std::atomic<uint64_t> count_barriers{0}, count_buffer_copies{0}, count_copy_regions{0}, count_draws{0},
    count_frames{0}, count_pipelines{0}, count_set_binds{0}, count_passes{0}, count_dispatches{0}, count_pushes{0},
    count_dynamic_state{0};
const bool counting = [] {
  const char* v = std::getenv("MCLA_VR_COUNT");
  return v && v[0] == '1';
}();
// Development (MCLA_VR_COUNT=1): cycles in the wrappers the GPU thread calls for every draw, past the driver's own.
enum WrapTimer { kTimeAllocate, kTimeUpdate, kTimeUpdateDriver, kTimeOther, kTimeUpdateSlow, kTimeCount };
std::atomic<uint64_t> wrap_cycles[kTimeCount];
std::atomic<uint64_t> wrap_calls[kTimeCount];
struct WrapTime {
  WrapTimer timer;
  uint64_t start;
  explicit WrapTime(WrapTimer t) : timer(t), start(counting ? __rdtsc() : 0) {}
  ~WrapTime() {
    if (counting) {
      wrap_cycles[timer] += __rdtsc() - start;
      ++wrap_calls[timer];
    }
  }
};

// The right eye's replay: each handle's twin looked up once per replay (the handles a submission uses outlive it),
// then from this thread's cache, so the lock is taken once per handle rather than once per use.
thread_local uint32_t replay_generation = 1;
template <typename K>
struct ReplayCacheEntry {
  K twin = VK_NULL_HANDLE;
  uint32_t generation = 0;
};
template <typename Map>
FlatMap<typename Map::key_type, ReplayCacheEntry<typename Map::key_type>>& ReplayCache(const Map&) {
  thread_local FlatMap<typename Map::key_type, ReplayCacheEntry<typename Map::key_type>> cache;
  return cache;
}

// A read of the maps: the GPU thread needs no lock.
std::shared_lock<std::shared_mutex> ReadLock() {
  return gpu_thread ? std::shared_lock<std::shared_mutex>{mutex, std::defer_lock}
                    : std::shared_lock<std::shared_mutex>{mutex};
}

template <typename Map>
typename Map::mapped_type Find(const Map& map, typename Map::key_type key) {
  if (key == VK_NULL_HANDLE) {
    return key;
  }
  if (replay_eye) {
    auto& entry = ReplayCache(map)[key];
    if (entry.generation != replay_generation) {
      std::shared_lock lock{mutex};
      const auto it = map.find(key);
      entry.twin = it != map.end() ? it->second : key;
      entry.generation = replay_generation;
    }
    return entry.twin;
  }
  auto lock = ReadLock();
  const auto it = map.find(key);
  return it != map.end() ? it->second : key;
}

void Log(const char* text) { REXLOG_INFO("VR: {}", text); }

// Work for the GPU thread from another thread (a destroyed resource's twin to forget), done at the GPU thread's next
// resource call.
void Foreign(std::function<void()> work) {
  std::scoped_lock lock{foreign_mutex};
  foreign_work.push_back(std::move(work));
  foreign_pending.store(true, std::memory_order_release);
}

void DrainForeign() {
  if (!foreign_pending.load(std::memory_order_acquire)) {
    return;
  }
  std::vector<std::function<void()>> work;
  {
    std::scoped_lock lock{foreign_mutex};
    work.swap(foreign_work);
    foreign_pending.store(false, std::memory_order_relaxed);
  }
  for (const auto& w : work) {
    w();
  }
}

// --- Memory ---------------------------------------------------------------------------------------------------------

VkResult VKAPI_PTR AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                                  const VkAllocationCallbacks* allocator, VkDeviceMemory* memory) {
  WrapTime time{kTimeOther};
  const VkResult result = real.vkAllocateMemory(device, info, allocator, memory);
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  if (result != VK_SUCCESS || ((host_visible_types >> info->memoryTypeIndex) & 1)) {
    return result;
  }
  VkMemoryAllocateInfo twin_info = *info;
  VkMemoryDedicatedAllocateInfo dedicated;
  VkMemoryAllocateFlagsInfo flags;
  const void** link = &twin_info.pNext;
  for (auto* p = static_cast<const VkBaseInStructure*>(info->pNext); p; p = p->pNext) {
    if (p->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO) {
      dedicated = *reinterpret_cast<const VkMemoryDedicatedAllocateInfo*>(p);
      auto lock = ReadLock();
      if (dedicated.image) {
        const auto it = pending_images.find(dedicated.image);
        if (it == pending_images.end()) {
          return result;
        }
        dedicated.image = it->second;
      }
      if (dedicated.buffer) {
        const auto it = pending_buffers.find(dedicated.buffer);
        if (it == pending_buffers.end()) {
          return result;
        }
        dedicated.buffer = it->second;
      }
      *link = &dedicated;
      link = &dedicated.pNext;
    } else if (p->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO) {
      flags = *reinterpret_cast<const VkMemoryAllocateFlagsInfo*>(p);
      *link = &flags;
      link = &flags.pNext;
    } else {
      static bool logged = false;
      if (!logged) {
        logged = true;
        REXLOG_WARN("VR: a device allocation with an unknown extension structure ({}): no right-eye twin",
                    int(p->sType));
      }
      return result;
    }
  }
  *link = nullptr;
  VkDeviceMemory twin;
  if (real.vkAllocateMemory(device, &twin_info, allocator, &twin) != VK_SUCCESS) {
    REXLOG_WARN("VR: no memory for a right-eye twin of {} bytes", info->allocationSize);
    return result;
  }
  twin_bytes += info->allocationSize;
  std::unique_lock lock{mutex};
  memory_twins[*memory] = twin;
  return result;
}

void ForgetMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{mutex};
  if (const auto it = memory_twins.find(memory); it != memory_twins.end()) {
    real.vkFreeMemory(device, it->second, allocator);
    memory_twins.erase(it);
  }
}

void VKAPI_PTR FreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator) {
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetMemory(device, memory, allocator); });
    real.vkFreeMemory(device, memory, allocator);
    return;
  }
  real.vkFreeMemory(device, memory, allocator);
  ForgetMemory(device, memory, allocator);
}

// --- Images and buffers: a twin made with the resource, kept once it is bound to twinned memory --------------------

VkResult VKAPI_PTR CreateImage(VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* allocator,
                               VkImage* image) {
  WrapTime time{kTimeOther};
  if (gpu_thread && std::exchange(layered_next, false) && info->arrayLayers == 1) {
    // Multiview: a render target, a layer per eye and no twin (its second layer is the right eye's).
    DrainForeign();
    VkImageCreateInfo layered = *info;
    layered.arrayLayers = 2;
    const VkResult result = real.vkCreateImage(device, &layered, allocator, image);
    if (result == VK_SUCCESS) {
      std::unique_lock lock{mutex};
      layered_images[*image] = 1;
      image_records[*image] = {};
    }
    static std::atomic<int> logged{0};
    if (logged++ < 3) {
      REXLOG_INFO("VR multiview: a layered render target {}x{} ({})", info->extent.width, info->extent.height,
                  int(result));
    }
    return result;
  }
  const VkResult result = real.vkCreateImage(device, info, allocator, image);
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  VkImage twin;
  if (result == VK_SUCCESS && real.vkCreateImage(device, info, allocator, &twin) == VK_SUCCESS) {
    std::unique_lock lock{mutex};
    pending_images[*image] = twin;
  }
  return result;
}

void BindImageTwin(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
  if (!gpu_thread) {
    return;
  }
  std::unique_lock lock{mutex};
  const auto pending = pending_images.find(image);
  if (pending == pending_images.end()) {
    return;
  }
  const VkImage twin = pending->second;
  pending_images.erase(pending);
  const auto twin_memory = memory_twins.find(memory);
  if (twin_memory != memory_twins.end() &&
      real.vkBindImageMemory(device, twin, twin_memory->second, offset) == VK_SUCCESS) {
    image_twins[image] = twin;
    image_records[image] = {};
  } else {
    real.vkDestroyImage(device, twin, nullptr);
  }
}

VkResult VKAPI_PTR BindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
  WrapTime time{kTimeOther};
  const VkResult result = real.vkBindImageMemory(device, image, memory, offset);
  if (result == VK_SUCCESS) {
    BindImageTwin(device, image, memory, offset);
  }
  return result;
}

VkResult VKAPI_PTR BindImageMemory2(VkDevice device, uint32_t count, const VkBindImageMemoryInfo* infos) {
  const VkResult result = real.vkBindImageMemory2(device, count, infos);
  if (result == VK_SUCCESS) {
    for (uint32_t i = 0; i < count; ++i) {
      BindImageTwin(device, infos[i].image, infos[i].memory, infos[i].memoryOffset);
    }
  }
  return result;
}

void ForgetImage(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{mutex};
  image_records.erase(image);
  layered_images.erase(image);
  for (auto* map : {&pending_images, &image_twins}) {
    if (const auto it = map->find(image); it != map->end()) {
      real.vkDestroyImage(device, it->second, allocator);
      map->erase(it);
    }
  }
}

void VKAPI_PTR DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator) {
  WrapTime time{kTimeOther};
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetImage(device, image, allocator); });
    real.vkDestroyImage(device, image, allocator);
    return;
  }
  real.vkDestroyImage(device, image, allocator);
  ForgetImage(device, image, allocator);
}

VkResult VKAPI_PTR CreateBuffer(VkDevice device, const VkBufferCreateInfo* info,
                                const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
  WrapTime time{kTimeOther};
  const VkResult result = real.vkCreateBuffer(device, info, allocator, buffer);
  const bool sparse_twin = sparse_twin_next && (info->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT);
  if (info->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) {
    sparse_twin_next = false;
  }
  if (sparse_twin && result == VK_SUCCESS && Stereo()) {  // whichever thread creates it
    VkBuffer twin;
    if (real.vkCreateBuffer(device, info, allocator, &twin) == VK_SUCCESS) {
      std::unique_lock lock{mutex};
      buffer_twins[*buffer] = twin;
      REXLOG_INFO("VR: a sparse buffer of {} MB with a sparse right-eye twin", info->size >> 20);
    }
    return result;
  }
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  if (result != VK_SUCCESS) {
    return result;
  }
  if (info->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) {
    // Others than SparseTwinNext's both eyes share (the shared memory).
    REXLOG_WARN("VR: a sparse buffer of {} bytes has no right-eye twin (both eyes share it)", info->size);
    return result;
  }
  VkBuffer twin;
  if (real.vkCreateBuffer(device, info, allocator, &twin) == VK_SUCCESS) {
    std::unique_lock lock{mutex};
    pending_buffers[*buffer] = twin;
  }
  return result;
}

void BindBufferTwin(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
  if (!gpu_thread) {
    return;
  }
  std::unique_lock lock{mutex};
  const auto pending = pending_buffers.find(buffer);
  if (pending == pending_buffers.end()) {
    return;
  }
  const VkBuffer twin = pending->second;
  pending_buffers.erase(pending);
  const auto twin_memory = memory_twins.find(memory);
  if (twin_memory != memory_twins.end() &&
      real.vkBindBufferMemory(device, twin, twin_memory->second, offset) == VK_SUCCESS) {
    buffer_twins[buffer] = twin;
  } else {
    real.vkDestroyBuffer(device, twin, nullptr);
  }
}

VkResult VKAPI_PTR BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
  const VkResult result = real.vkBindBufferMemory(device, buffer, memory, offset);
  if (result == VK_SUCCESS) {
    BindBufferTwin(device, buffer, memory, offset);
  }
  return result;
}

VkResult VKAPI_PTR BindBufferMemory2(VkDevice device, uint32_t count, const VkBindBufferMemoryInfo* infos) {
  const VkResult result = real.vkBindBufferMemory2(device, count, infos);
  if (result == VK_SUCCESS) {
    for (uint32_t i = 0; i < count; ++i) {
      BindBufferTwin(device, infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
    }
  }
  return result;
}

void ForgetBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{mutex};
  for (auto* map : {&pending_buffers, &buffer_twins}) {
    if (const auto it = map->find(buffer); it != map->end()) {
      real.vkDestroyBuffer(device, it->second, allocator);
      map->erase(it);
    }
  }
}

void VKAPI_PTR DestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator) {
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetBuffer(device, buffer, allocator); });
    real.vkDestroyBuffer(device, buffer, allocator);
    return;
  }
  real.vkDestroyBuffer(device, buffer, allocator);
  ForgetBuffer(device, buffer, allocator);
}

// --- Views, framebuffers, query pools --------------------------------------------------------------------------------

VkResult VKAPI_PTR CreateImageView(VkDevice device, const VkImageViewCreateInfo* info,
                                   const VkAllocationCallbacks* allocator, VkImageView* view) {
  WrapTime time{kTimeOther};
  if (gpu_thread && layered_images.count(info->image)) {
    DrainForeign();
    VkImageViewCreateInfo eye_info = *info;
    eye_info.subresourceRange.baseArrayLayer = 0;
    eye_info.subresourceRange.layerCount = 1;
    const VkResult result = real.vkCreateImageView(device, &eye_info, allocator, view);
    if (result != VK_SUCCESS) {
      return result;
    }
    eye_info.subresourceRange.baseArrayLayer = 1;
    VkImageView twin = VK_NULL_HANDLE, both = VK_NULL_HANDLE;
    VkImageViewCreateInfo both_info = *info;
    both_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    both_info.subresourceRange.baseArrayLayer = 0;
    both_info.subresourceRange.layerCount = 2;
    if (real.vkCreateImageView(device, &eye_info, allocator, &twin) == VK_SUCCESS &&
        real.vkCreateImageView(device, &both_info, allocator, &both) == VK_SUCCESS) {
      std::unique_lock lock{mutex};
      image_view_twins[*view] = twin;
      const auto record = image_records.find(info->image);
      view_records[*view] = {twin, record != image_records.end() ? &record->second : nullptr};
      array_views[*view] = both;
    } else {
      REXLOG_WARN("VR multiview: no eye views of a render target");
    }
    return result;
  }
  const VkResult result = real.vkCreateImageView(device, info, allocator, view);
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  const VkImage twin_image = Find(image_twins, info->image);
  if (result == VK_SUCCESS && twin_image != info->image) {
    VkImageViewCreateInfo twin_info = *info;
    twin_info.image = twin_image;
    VkImageView twin;
    if (real.vkCreateImageView(device, &twin_info, allocator, &twin) == VK_SUCCESS) {
      std::unique_lock lock{mutex};
      image_view_twins[*view] = twin;
      const auto record = image_records.find(info->image);
      view_records[*view] = {twin, record != image_records.end() ? &record->second : nullptr};
    }
  }
  return result;
}

void ForgetImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{mutex};
  if (const auto it = array_views.find(view); it != array_views.end()) {
    real.vkDestroyImageView(device, it->second, allocator);
    array_views.erase(it);
  }
  if (const auto it = image_view_twins.find(view); it != image_view_twins.end()) {
    real.vkDestroyImageView(device, it->second, allocator);
    image_view_twins.erase(it);
    view_records.erase(view);
  }
}

void VKAPI_PTR DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator) {
  WrapTime time{kTimeOther};
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetImageView(device, view, allocator); });
    real.vkDestroyImageView(device, view, allocator);
    return;
  }
  real.vkDestroyImageView(device, view, allocator);
  ForgetImageView(device, view, allocator);
}

VkResult VKAPI_PTR CreateBufferView(VkDevice device, const VkBufferViewCreateInfo* info,
                                    const VkAllocationCallbacks* allocator, VkBufferView* view) {
  const VkResult result = real.vkCreateBufferView(device, info, allocator, view);
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  const VkBuffer twin_buffer = Find(buffer_twins, info->buffer);
  if (result == VK_SUCCESS && twin_buffer != info->buffer) {
    VkBufferViewCreateInfo twin_info = *info;
    twin_info.buffer = twin_buffer;
    VkBufferView twin;
    if (real.vkCreateBufferView(device, &twin_info, allocator, &twin) == VK_SUCCESS) {
      std::unique_lock lock{mutex};
      buffer_view_twins[*view] = twin;
    }
  }
  return result;
}

void ForgetBufferView(VkDevice device, VkBufferView view, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{mutex};
  if (const auto it = buffer_view_twins.find(view); it != buffer_view_twins.end()) {
    real.vkDestroyBufferView(device, it->second, allocator);
    buffer_view_twins.erase(it);
  }
}

void VKAPI_PTR DestroyBufferView(VkDevice device, VkBufferView view, const VkAllocationCallbacks* allocator) {
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetBufferView(device, view, allocator); });
    real.vkDestroyBufferView(device, view, allocator);
    return;
  }
  real.vkDestroyBufferView(device, view, allocator);
  ForgetBufferView(device, view, allocator);
}

VkResult VKAPI_PTR CreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* info,
                                     const VkAllocationCallbacks* allocator, VkFramebuffer* framebuffer) {
  WrapTime time{kTimeOther};
  const VkResult result = real.vkCreateFramebuffer(device, info, allocator, framebuffer);
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  if (result != VK_SUCCESS || (info->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT)) {
    return result;
  }
  std::vector<VkImageView> attachments(info->pAttachments, info->pAttachments + info->attachmentCount);
  bool twinned = false;
  for (VkImageView& view : attachments) {
    const VkImageView twin = Find(image_view_twins, view);
    twinned |= twin != view;
    view = twin;
  }
  if (twinned) {
    VkFramebufferCreateInfo twin_info = *info;
    twin_info.pAttachments = attachments.data();
    VkFramebuffer twin;
    if (real.vkCreateFramebuffer(device, &twin_info, allocator, &twin) == VK_SUCCESS) {
      std::unique_lock lock{mutex};
      framebuffer_twins[*framebuffer] = twin;
    }
  }
  return result;
}

void ForgetFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{mutex};
  if (const auto it = framebuffer_twins.find(framebuffer); it != framebuffer_twins.end()) {
    real.vkDestroyFramebuffer(device, it->second, allocator);
    framebuffer_twins.erase(it);
  }
}

void VKAPI_PTR DestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* allocator) {
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetFramebuffer(device, framebuffer, allocator); });
    real.vkDestroyFramebuffer(device, framebuffer, allocator);
    return;
  }
  real.vkDestroyFramebuffer(device, framebuffer, allocator);
  ForgetFramebuffer(device, framebuffer, allocator);
}

VkResult VKAPI_PTR CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* info,
                                   const VkAllocationCallbacks* allocator, VkQueryPool* pool) {
  const VkResult result = real.vkCreateQueryPool(device, info, allocator, pool);
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  VkQueryPool twin;
  if (result == VK_SUCCESS && real.vkCreateQueryPool(device, info, allocator, &twin) == VK_SUCCESS) {
    std::unique_lock lock{mutex};
    query_pool_twins[*pool] = twin;
  }
  return result;
}

void ForgetQueryPool(VkDevice device, VkQueryPool pool, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{mutex};
  if (const auto it = query_pool_twins.find(pool); it != query_pool_twins.end()) {
    real.vkDestroyQueryPool(device, it->second, allocator);
    query_pool_twins.erase(it);
  }
}

void VKAPI_PTR DestroyQueryPool(VkDevice device, VkQueryPool pool, const VkAllocationCallbacks* allocator) {
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetQueryPool(device, pool, allocator); });
    real.vkDestroyQueryPool(device, pool, allocator);
    return;
  }
  real.vkDestroyQueryPool(device, pool, allocator);
  ForgetQueryPool(device, pool, allocator);
}

// --- Descriptors: a set's twin is made when a write puts an eye-dependent resource in it ------------------------------

VkResult VKAPI_PTR CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* info,
                                        const VkAllocationCallbacks* allocator, VkDescriptorPool* pool) {
  const VkResult result = real.vkCreateDescriptorPool(device, info, allocator, pool);
  if (!gpu_thread) {
    return result;
  }
  DrainForeign();
  VkDescriptorPool twin;
  if (result == VK_SUCCESS && real.vkCreateDescriptorPool(device, info, allocator, &twin) == VK_SUCCESS) {
    std::unique_lock lock{set_mutex};
    pool_twins[*pool] = twin;
  }
  return result;
}

void ForgetPoolSets(VkDescriptorPool pool) {
  if (const auto it = pool_sets.find(pool); it != pool_sets.end()) {
    for (VkDescriptorSet set : it->second) {
      set_twins.erase(set);
      dormant_twins.erase(set);
    }
    pool_sets.erase(it);
  }
  if (const auto it = lasting_pool_sets.find(pool); it != lasting_pool_sets.end()) {
    for (VkDescriptorSet set : it->second) {
      lasting_sets.erase(set);
    }
    lasting_pool_sets.erase(it);
  }
}

void ForgetDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* allocator) {
  std::unique_lock lock{set_mutex};
  if (const auto it = pool_twins.find(pool); it != pool_twins.end()) {
    real.vkDestroyDescriptorPool(device, it->second, allocator);
    pool_twins.erase(it);
  }
  ForgetPoolSets(pool);
  reset_pools.erase(pool);
}

void VKAPI_PTR DestroyDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* allocator) {
  if (!gpu_thread) {  // queued first: the handle is not reused before it is destroyed
    Foreign([=] { ForgetDescriptorPool(device, pool, allocator); });
    real.vkDestroyDescriptorPool(device, pool, allocator);
    return;
  }
  real.vkDestroyDescriptorPool(device, pool, allocator);
  ForgetDescriptorPool(device, pool, allocator);
}

void ForgetPoolReset(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags) {
  std::unique_lock lock{set_mutex};
  if (const auto it = pool_twins.find(pool); it != pool_twins.end()) {
    real.vkResetDescriptorPool(device, it->second, flags);
  }
  ForgetPoolSets(pool);
  reset_pools.insert(pool);
}

VkResult VKAPI_PTR ResetDescriptorPool(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags) {
  const VkResult result = real.vkResetDescriptorPool(device, pool, flags);
  if (!gpu_thread) {
    Foreign([=] { ForgetPoolReset(device, pool, flags); });
    return result;
  }
  ForgetPoolReset(device, pool, flags);
  return result;
}

VkResult VKAPI_PTR AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* info,
                                          VkDescriptorSet* sets) {
  WrapTime time{kTimeAllocate};
  const VkResult result = real.vkAllocateDescriptorSets(device, info, sets);
  if (result != VK_SUCCESS || info->pNext || !gpu_thread) {
    return result;
  }
  for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
    recent_sets[recent_next] = {sets[i], info->descriptorPool, info->pSetLayouts[i]};
    recent_next = (recent_next + 1) % 64;
  }
  sets_made += info->descriptorSetCount;
  thread_local VkDescriptorPool last_reset_pool = VK_NULL_HANDLE;
  if (info->descriptorPool == last_reset_pool) {
    return result;
  }
  {
    auto lock = ReadLock();
    if (reset_pools.count(info->descriptorPool)) {
      last_reset_pool = info->descriptorPool;
      return result;
    }
  }
  std::unique_lock lock{set_mutex};
  auto& pool_list = lasting_pool_sets[info->descriptorPool];
  for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
    lasting_sets[sets[i]] = {sets[i], info->descriptorPool, info->pSetLayouts[i]};
    pool_list.push_back(sets[i]);
  }
  return result;
}

// The set's twin, made now if `make` (set_mutex held exclusively); VK_NULL_HANDLE if it has none.
VkDescriptorSet SetTwin(VkDevice device, VkDescriptorSet set, bool make) {
  if (const auto it = set_twins.find(set); it != set_twins.end()) {
    return it->second;
  }
  if (!make) {
    return VK_NULL_HANDLE;
  }
  if (const auto it = dormant_twins.find(set); it != dormant_twins.end()) {
    const VkDescriptorSet twin = it->second;
    dormant_twins.erase(it);
    set_twins[set] = twin;
    return twin;
  }
  const RecentSet* recent = nullptr;
  for (uint32_t i = 1; i <= 64 && !recent; ++i) {
    const RecentSet& candidate = recent_sets[(recent_next + 64 - i) % 64];
    recent = candidate.set == set ? &candidate : nullptr;
  }
  if (!recent) {
    const auto lasting = lasting_sets.find(set);
    recent = lasting != lasting_sets.end() ? &lasting->second : nullptr;
  }
  const auto pool_twin = recent ? pool_twins.find(recent->pool) : pool_twins.end();
  if (pool_twin == pool_twins.end()) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXLOG_WARN("VR: an eye-dependent write to a descriptor set made long before (or elsewhere): no twin");
    }
    return VK_NULL_HANDLE;
  }
  VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate.descriptorPool = pool_twin->second;
  allocate.descriptorSetCount = 1;
  allocate.pSetLayouts = &recent->layout;
  VkDescriptorSet twin;
  if (real.vkAllocateDescriptorSets(device, &allocate, &twin) != VK_SUCCESS) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXLOG_WARN("VR: no right-eye twin for a descriptor set");
    }
    return VK_NULL_HANDLE;
  }
  set_twins[set] = twin;
  pool_sets[recent->pool].push_back(set);
  return twin;
}

// The right eye's view: the twin, unless the image holds the same picture in both eyes. (mutex held.)
VkImageView EyeView(VkImageView view) {
  const auto record = view_records.find(view);
  if (record == view_records.end()) {
    return view;
  }
  return record->second.image && record->second.image->eye_independent ? view : record->second.twin;
}

// Whether a write puts an eye-dependent resource in its set. (mutex held.)
bool WriteIsEyeDependent(const VkWriteDescriptorSet& write) {
  for (uint32_t j = 0; j < write.descriptorCount; ++j) {
    if ((write.pImageInfo && write.pImageInfo[j].imageView &&
         EyeView(write.pImageInfo[j].imageView) != write.pImageInfo[j].imageView) ||
        (write.pBufferInfo && write.pBufferInfo[j].buffer && buffer_twins.count(write.pBufferInfo[j].buffer)) ||
        (write.pTexelBufferView && write.pTexelBufferView[j] &&
         buffer_view_twins.count(write.pTexelBufferView[j]))) {
      return true;
    }
  }
  return false;
}

void VKAPI_PTR UpdateDescriptorSets(VkDevice device, uint32_t write_count, const VkWriteDescriptorSet* writes,
                                    uint32_t copy_count, const VkCopyDescriptorSet* copies) {
  WrapTime time{kTimeUpdate};
  {
    WrapTime driver{kTimeUpdateDriver};
    real.vkUpdateDescriptorSets(device, write_count, writes, copy_count, copies);
  }
  if (!gpu_thread) {
    return;
  }
  {
    // Most writes (textures the same in both eyes, to sets without twins) need nothing more.
    bool any = copy_count != 0;
    {
      auto lock = ReadLock();
      for (uint32_t i = 0; i < write_count && !any; ++i) {
        any = WriteIsEyeDependent(writes[i]) || set_twins.count(writes[i].dstSet) != 0;
      }
    }
    if (!any) {
      return;
    }
  }
  WrapTime slow{kTimeUpdateSlow};
  thread_local std::vector<VkWriteDescriptorSet> twin_writes;
  thread_local std::vector<VkCopyDescriptorSet> twin_copies;
  thread_local std::vector<VkDescriptorImageInfo> image_infos;
  thread_local std::vector<VkDescriptorBufferInfo> buffer_infos;
  thread_local std::vector<VkBufferView> texel_views;
  thread_local std::vector<uint8_t> dependent;  // per write: it puts an eye-dependent resource in its set
  twin_writes.clear();
  twin_copies.clear();
  size_t images = 0, buffers = 0, texels = 0;
  for (uint32_t i = 0; i < write_count; ++i) {
    images += writes[i].pImageInfo ? writes[i].descriptorCount : 0;
    buffers += writes[i].pBufferInfo ? writes[i].descriptorCount : 0;
    texels += writes[i].pTexelBufferView ? writes[i].descriptorCount : 0;
  }
  image_infos.resize(images);
  buffer_infos.resize(buffers);
  texel_views.resize(texels);
  dependent.assign(write_count, 0);
  images = buffers = texels = 0;
  {
    auto lock = ReadLock();
    const auto twin = [](const auto& map, auto key) {
      const auto it = map.find(key);
      return it != map.end() ? it->second : key;
    };
    for (uint32_t i = 0; i < write_count; ++i) {
      VkWriteDescriptorSet write = writes[i];
      bool eye_dependent = false;
      if (write.pImageInfo) {
        for (uint32_t j = 0; j < write.descriptorCount; ++j) {
          VkDescriptorImageInfo info = write.pImageInfo[j];
          if (info.imageView) {
            const VkImageView eye_view = EyeView(info.imageView);
            eye_dependent |= eye_view != info.imageView;
            info.imageView = eye_view;
          }
          image_infos[images + j] = info;
        }
        write.pImageInfo = &image_infos[images];
        images += write.descriptorCount;
      }
      if (write.pBufferInfo) {
        for (uint32_t j = 0; j < write.descriptorCount; ++j) {
          VkDescriptorBufferInfo info = write.pBufferInfo[j];
          if (info.buffer) {
            const VkBuffer eye_buffer = twin(buffer_twins, info.buffer);
            eye_dependent |= eye_buffer != info.buffer;
            info.buffer = eye_buffer;
          }
          buffer_infos[buffers + j] = info;
        }
        write.pBufferInfo = &buffer_infos[buffers];
        buffers += write.descriptorCount;
      }
      if (write.pTexelBufferView) {
        for (uint32_t j = 0; j < write.descriptorCount; ++j) {
          VkBufferView view = write.pTexelBufferView[j];
          if (view) {
            const VkBufferView eye_view = twin(buffer_view_twins, view);
            eye_dependent |= eye_view != view;
            view = eye_view;
          }
          texel_views[texels + j] = view;
        }
        write.pTexelBufferView = &texel_views[texels];
        texels += write.descriptorCount;
      }
      dependent[i] = eye_dependent;
      twin_writes.push_back(write);
    }
  }
  // Each set written: its twin (made if a write to it is eye-dependent) gets all of this call's writes to it; a set
  // written with nothing eye-dependent (the GPU thread writes whole sets) lets its twin sleep.
  std::unique_lock lock{set_mutex};
  size_t kept = 0;
  for (uint32_t i = 0; i < write_count; ++i) {
    bool make = false;
    for (uint32_t k = 0; k < write_count && !make; ++k) {
      make = dependent[k] && writes[k].dstSet == writes[i].dstSet;
    }
    if (!make) {
      if (const auto it = set_twins.find(writes[i].dstSet); it != set_twins.end()) {
        dormant_twins[writes[i].dstSet] = it->second;
        set_twins.erase(it);
      }
      continue;
    }
    const VkDescriptorSet set_twin = SetTwin(device, writes[i].dstSet, true);
    if (set_twin) {
      twin_writes[kept] = twin_writes[i];
      twin_writes[kept].dstSet = set_twin;
      ++kept;
    }
  }
  twin_writes.resize(kept);
  for (uint32_t i = 0; i < copy_count; ++i) {
    const VkDescriptorSet src = SetTwin(device, copies[i].srcSet, false);
    const VkDescriptorSet dst = SetTwin(device, copies[i].dstSet, src != VK_NULL_HANDLE);
    if (dst) {
      VkCopyDescriptorSet copy = copies[i];
      copy.srcSet = src ? src : copies[i].srcSet;
      copy.dstSet = dst;
      twin_copies.push_back(copy);
    }
  }
  if (!twin_writes.empty() || !twin_copies.empty()) {
    real.vkUpdateDescriptorSets(device, uint32_t(twin_writes.size()), twin_writes.data(),
                                uint32_t(twin_copies.size()), twin_copies.data());
  }
}

// --- The right eye's command functions -------------------------------------------------------------------------------

void VKAPI_PTR RightBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bind_point, VkPipeline pipeline) {
  if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS && RightSkipsGraphics()) {
    return;
  }
  ++count_pipelines;
  real.vkCmdBindPipeline(cb, bind_point, pipeline);
}
void VKAPI_PTR RightDraw(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  ++count_draws;
  real.vkCmdDraw(cb, a, b, c, d);
}
void VKAPI_PTR RightDrawIndexed(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e) {
  ++count_draws;
  real.vkCmdDrawIndexed(cb, a, b, c, d, e);
}
void VKAPI_PTR RightDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
  ++count_dispatches;
  real.vkCmdDispatch(cb, x, y, z);
}
void VKAPI_PTR RightPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages,
                                  uint32_t offset, uint32_t size, const void* values) {
  if (!(stages & VK_SHADER_STAGE_COMPUTE_BIT) && RightSkipsGraphics()) {
    return;
  }
  ++count_pushes;
  real.vkCmdPushConstants(cb, layout, stages, offset, size, values);
}
void VKAPI_PTR RightSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkViewport* v) {
  ++count_dynamic_state;
  real.vkCmdSetViewport(cb, first, count, v);
}
void VKAPI_PTR RightSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkRect2D* r) {
  ++count_dynamic_state;
  real.vkCmdSetScissor(cb, first, count, r);
}

void VKAPI_PTR RightBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* info, VkSubpassContents contents) {
  right_in_pass = true;
  ++count_passes;
  VkRenderPassBeginInfo twin = *info;
  twin.framebuffer = Find(framebuffer_twins, info->framebuffer);
  real.vkCmdBeginRenderPass(cb, &twin, contents);
}

void VKAPI_PTR RightEndRenderPass(VkCommandBuffer cb) {
  right_in_pass = false;
  real.vkCmdEndRenderPass(cb);
}

void VKAPI_PTR RightEndRendering(VkCommandBuffer cb) {
  right_in_pass = false;
  real.vkCmdEndRendering(cb);
}

void VKAPI_PTR RightBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
  right_in_pass = true;
  ++count_passes;
  VkRenderingInfo twin = *info;
  VkRenderingAttachmentInfo colors[16];
  const uint32_t count = std::min(info->colorAttachmentCount, uint32_t(16));
  for (uint32_t i = 0; i < count; ++i) {
    colors[i] = info->pColorAttachments[i];
    colors[i].imageView = Find(image_view_twins, colors[i].imageView);
    colors[i].resolveImageView = Find(image_view_twins, colors[i].resolveImageView);
  }
  twin.pColorAttachments = count ? colors : nullptr;
  VkRenderingAttachmentInfo depth, stencil;
  if (info->pDepthAttachment) {
    depth = *info->pDepthAttachment;
    depth.imageView = Find(image_view_twins, depth.imageView);
    depth.resolveImageView = Find(image_view_twins, depth.resolveImageView);
    twin.pDepthAttachment = &depth;
  }
  if (info->pStencilAttachment) {
    stencil = *info->pStencilAttachment;
    stencil.imageView = Find(image_view_twins, stencil.imageView);
    stencil.resolveImageView = Find(image_view_twins, stencil.resolveImageView);
    twin.pStencilAttachment = &stencil;
  }
  real.vkCmdBeginRendering(cb, &twin);
}

void VKAPI_PTR RightBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                       uint32_t first, uint32_t count, const VkDescriptorSet* sets,
                                       uint32_t dynamic_count, const uint32_t* dynamic_offsets) {
  if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS && RightSkipsGraphics()) {
    return;
  }
  ++count_set_binds;
  VkDescriptorSet twins[16];
  const uint32_t n = std::min(count, uint32_t(16));
  for (uint32_t i = 0; i < n; ++i) {
    twins[i] = Find(set_twins, sets[i]);
  }
  real.vkCmdBindDescriptorSets(cb, bind_point, layout, first, n, twins, dynamic_count, dynamic_offsets);
}

void VKAPI_PTR RightBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
  if (RightSkipsGraphics()) {
    return;
  }
  real.vkCmdBindIndexBuffer(cb, Find(buffer_twins, buffer), offset, type);
}

void VKAPI_PTR RightBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* buffers,
                                      const VkDeviceSize* offsets) {
  if (RightSkipsGraphics()) {
    return;
  }
  VkBuffer twins[32];
  const uint32_t n = std::min(count, uint32_t(32));
  for (uint32_t i = 0; i < n; ++i) {
    twins[i] = Find(buffer_twins, buffers[i]);
  }
  real.vkCmdBindVertexBuffers(cb, first, n, twins, offsets);
}

void VKAPI_PTR RightClearColorImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
                                    const VkClearColorValue* color, uint32_t count,
                                    const VkImageSubresourceRange* ranges) {
  real.vkCmdClearColorImage(cb, Find(image_twins, image), layout, color, count, ranges);
}

void VKAPI_PTR RightCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t count,
                               const VkBufferCopy* regions) {
  if (counting) {
    ++count_buffer_copies;
    count_copy_regions += count;
  }
  real.vkCmdCopyBuffer(cb, Find(buffer_twins, src), Find(buffer_twins, dst), count, regions);
}

// A layered image's copies: the eye's layer.
const VkBufferImageCopy* EyeRegions(VkImage image, uint32_t count, const VkBufferImageCopy* regions) {
  if (!EyeLayer(image)) {
    return regions;
  }
  thread_local std::vector<VkBufferImageCopy> eye;
  eye.assign(regions, regions + count);
  for (auto& region : eye) {
    region.imageSubresource.baseArrayLayer = 1;
  }
  return eye.data();
}

void VKAPI_PTR RightCopyBufferToImage(VkCommandBuffer cb, VkBuffer buffer, VkImage image, VkImageLayout layout,
                                      uint32_t count, const VkBufferImageCopy* regions) {
  real.vkCmdCopyBufferToImage(cb, Find(buffer_twins, buffer), Find(image_twins, image), layout, count,
                              EyeRegions(image, count, regions));
}

void VKAPI_PTR RightCopyImageToBuffer(VkCommandBuffer cb, VkImage image, VkImageLayout layout, VkBuffer buffer,
                                      uint32_t count, const VkBufferImageCopy* regions) {
  real.vkCmdCopyImageToBuffer(cb, Find(image_twins, image), layout, Find(buffer_twins, buffer), count,
                              EyeRegions(image, count, regions));
}

void VKAPI_PTR RightCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count,
                                         VkBuffer buffer, VkDeviceSize offset, VkDeviceSize stride,
                                         VkQueryResultFlags flags) {
  if (Multiview()) {
    return;
  }
  real.vkCmdCopyQueryPoolResults(cb, Find(query_pool_twins, pool), first, count, Find(buffer_twins, buffer), offset,
                                 stride, flags);
}

void VKAPI_PTR RightBeginQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query, VkQueryControlFlags flags) {
  if (Multiview()) {
    return;
  }
  real.vkCmdBeginQuery(cb, Find(query_pool_twins, pool), query, flags);
}

void VKAPI_PTR RightEndQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query) {
  if (Multiview()) {
    return;
  }
  real.vkCmdEndQuery(cb, Find(query_pool_twins, pool), query);
}

void VKAPI_PTR RightResetQueryPool(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count) {
  if (Multiview()) {
    return;
  }
  real.vkCmdResetQueryPool(cb, Find(query_pool_twins, pool), first, count);
}

void VKAPI_PTR RightWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits stage, VkQueryPool pool,
                                   uint32_t query) {
  if (Multiview()) {
    return;
  }
  real.vkCmdWriteTimestamp(cb, stage, Find(query_pool_twins, pool), query);
}

void VKAPI_PTR RightPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                    VkDependencyFlags dependency, uint32_t memory_count,
                                    const VkMemoryBarrier* memory_barriers, uint32_t buffer_count,
                                    const VkBufferMemoryBarrier* buffer_barriers, uint32_t image_count,
                                    const VkImageMemoryBarrier* image_barriers) {
  if (counting) {
    ++count_barriers;
  }
  thread_local std::vector<VkBufferMemoryBarrier> buffers;
  thread_local std::vector<VkImageMemoryBarrier> images;
  buffers.assign(buffer_barriers, buffer_barriers + buffer_count);
  images.assign(image_barriers, image_barriers + image_count);
  for (auto& barrier : buffers) {
    barrier.buffer = Find(buffer_twins, barrier.buffer);
  }
  for (auto& barrier : images) {
    bool layered;
    {
      auto lock = ReadLock();
      layered = layered_images.count(barrier.image) != 0;
    }
    if (layered) {
      barrier.subresourceRange.baseArrayLayer = 1;
      barrier.subresourceRange.layerCount = 1;
    } else {
      barrier.image = Find(image_twins, barrier.image);
    }
  }
  real.vkCmdPipelineBarrier(cb, src, dst, dependency, memory_count, memory_barriers, buffer_count, buffers.data(),
                            image_count, images.data());
}

// --- Multiview: the left eye's command functions (and the passes drawing both eyes) -----------------------------------

void VKAPI_PTR LeftPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                   VkDependencyFlags dependency, uint32_t memory_count,
                                   const VkMemoryBarrier* memory_barriers, uint32_t buffer_count,
                                   const VkBufferMemoryBarrier* buffer_barriers, uint32_t image_count,
                                   const VkImageMemoryBarrier* image_barriers) {
  // Outside a pass drawing both eyes, the left eye's layer (the right eye's replay does its own); in one (none are
  // recorded there), both.
  thread_local std::vector<VkImageMemoryBarrier> images;
  auto lock = ReadLock();
  bool layered = false;
  for (uint32_t i = 0; i < image_count && !layered; ++i) {
    layered = layered_images.count(image_barriers[i].image) != 0;
  }
  if (layered && !multiview_pass) {
    images.assign(image_barriers, image_barriers + image_count);
    for (auto& barrier : images) {
      if (layered_images.count(barrier.image)) {
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
      }
    }
    image_barriers = images.data();
  }
  real.vkCmdPipelineBarrier(cb, src, dst, dependency, memory_count, memory_barriers, buffer_count, buffer_barriers,
                            image_count, image_barriers);
}

void VKAPI_PTR LeftBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
  // A pass whose attachments are all layered draws both eyes: their views of both layers, view mask 0b11.
  VkImageView both[16 + 2] = {};
  const uint32_t count = std::min(info->colorAttachmentCount, uint32_t(16));
  bool layered = true, any = false;
  auto lock = ReadLock();
  const auto both_of = [&](VkImageView view, VkImageView& out) {
    if (!view) {
      return;
    }
    any = true;
    const auto it = array_views.find(view);
    if (it == array_views.end()) {
      layered = false;
    } else {
      out = it->second;
    }
  };
  for (uint32_t i = 0; i < count; ++i) {
    both_of(info->pColorAttachments[i].imageView, both[i]);
  }
  if (info->pDepthAttachment) {
    both_of(info->pDepthAttachment->imageView, both[16]);
  }
  if (info->pStencilAttachment) {
    both_of(info->pStencilAttachment->imageView, both[17]);
  }
  {
    static std::atomic<uint64_t> passes_mv{0}, passes_plain{0};
    static std::atomic<int> logged{0};
    ++(layered && any ? passes_mv : passes_plain);
    if (!(layered && any) && logged < 12) {
      ++logged;
      const VkImageView first = count ? info->pColorAttachments[0].imageView
                                      : info->pDepthAttachment ? info->pDepthAttachment->imageView : VK_NULL_HANDLE;
      const auto record = view_records.find(first);
      REXLOG_INFO("VR multiview: a pass not drawing both eyes: {} colour, depth {}, stencil {}, any {}, first view {} "
                  "(twin {}), {}x{}",
                  count, info->pDepthAttachment != nullptr, info->pStencilAttachment != nullptr, any, (void*)first,
                  record != view_records.end() ? (void*)record->second.twin : nullptr, info->renderArea.extent.width,
                  info->renderArea.extent.height);
    }
    if ((passes_mv + passes_plain) % 300000 == 0) {
      REXLOG_INFO("VR multiview: passes drawing both eyes {}, per eye {}", passes_mv.load(), passes_plain.load());
    }
  }
  lock = {};
  multiview_pass = last_pass_multiview = layered && any;
  if (!multiview_pass) {
    real.vkCmdBeginRendering(cb, info);
    return;
  }
  VkRenderingInfo mv = *info;
  VkRenderingAttachmentInfo colors[16];
  for (uint32_t i = 0; i < count; ++i) {
    colors[i] = info->pColorAttachments[i];
    colors[i].imageView = both[i] ? both[i] : colors[i].imageView;
  }
  mv.pColorAttachments = count ? colors : nullptr;
  VkRenderingAttachmentInfo depth, stencil;
  if (info->pDepthAttachment) {
    depth = *info->pDepthAttachment;
    depth.imageView = both[16] ? both[16] : depth.imageView;
    mv.pDepthAttachment = &depth;
  }
  if (info->pStencilAttachment) {
    stencil = *info->pStencilAttachment;
    stencil.imageView = both[17] ? both[17] : stencil.imageView;
    mv.pStencilAttachment = &stencil;
  }
  mv.viewMask = 0b11;
  mv.layerCount = 1;
  real.vkCmdBeginRendering(cb, &mv);
}

void VKAPI_PTR LeftEndRendering(VkCommandBuffer cb) {
  multiview_pass = false;
  real.vkCmdEndRendering(cb);
}

void VKAPI_PTR LeftBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* info,
                                   VkSubpassContents contents) {
  multiview_pass = last_pass_multiview = false;  // (render passes: the presenter's own; multiview needs dynamic rendering)
  real.vkCmdBeginRenderPass(cb, info, contents);
}

void VKAPI_PTR LeftBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                      uint32_t first, uint32_t count, const VkDescriptorSet* sets,
                                      uint32_t dynamic_count, const uint32_t* dynamic_offsets) {
  if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
    VkDescriptorSet variants[16];
    const uint32_t n = std::min(count, uint32_t(16));
    auto lock = ReadLock();
    for (uint32_t i = 0; i < n; ++i) {
      const auto it = multiview_variants.find(sets[i]);
      variants[i] = it != multiview_variants.end() ? it->second : sets[i];
    }
    real.vkCmdBindDescriptorSets(cb, bind_point, layout, first, n, variants, dynamic_count, dynamic_offsets);
    return;
  }
  real.vkCmdBindDescriptorSets(cb, bind_point, layout, first, count, sets, dynamic_count, dynamic_offsets);
}

}  // namespace

bool Stereo() {
  static const bool wanted = [] {
    const char* v = std::getenv("MCLA_VR_STEREO");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return wanted && Active();
}

void InstallTwins(VulkanDevice::Functions& functions, uint32_t host_visible_memory_types) {
  if (!Stereo()) {
    return;
  }
  real = functions;
  host_visible_types = host_visible_memory_types;
  functions.vkAllocateMemory = AllocateMemory;
  functions.vkFreeMemory = FreeMemory;
  functions.vkCreateImage = CreateImage;
  functions.vkBindImageMemory = BindImageMemory;
  if (functions.vkBindImageMemory2) {
    functions.vkBindImageMemory2 = BindImageMemory2;
  }
  functions.vkDestroyImage = DestroyImage;
  functions.vkCreateBuffer = CreateBuffer;
  functions.vkBindBufferMemory = BindBufferMemory;
  if (functions.vkBindBufferMemory2) {
    functions.vkBindBufferMemory2 = BindBufferMemory2;
  }
  functions.vkDestroyBuffer = DestroyBuffer;
  functions.vkCreateImageView = CreateImageView;
  functions.vkDestroyImageView = DestroyImageView;
  functions.vkCreateBufferView = CreateBufferView;
  functions.vkDestroyBufferView = DestroyBufferView;
  functions.vkCreateFramebuffer = CreateFramebuffer;
  functions.vkDestroyFramebuffer = DestroyFramebuffer;
  functions.vkCreateQueryPool = CreateQueryPool;
  functions.vkDestroyQueryPool = DestroyQueryPool;
  functions.vkCreateDescriptorPool = CreateDescriptorPool;
  functions.vkDestroyDescriptorPool = DestroyDescriptorPool;
  functions.vkResetDescriptorPool = ResetDescriptorPool;
  functions.vkAllocateDescriptorSets = AllocateDescriptorSets;
  functions.vkUpdateDescriptorSets = UpdateDescriptorSets;

  right = functions;
  right.vkCmdBeginRenderPass = RightBeginRenderPass;
  right.vkCmdEndRenderPass = RightEndRenderPass;
  if (right.vkCmdEndRendering) {
    right.vkCmdEndRendering = RightEndRendering;
  }
  if (counting || Multiview()) {
    right.vkCmdBindPipeline = RightBindPipeline;
    right.vkCmdPushConstants = RightPushConstants;
  }
  if (counting) {
    right.vkCmdDraw = RightDraw;
    right.vkCmdDrawIndexed = RightDrawIndexed;
    right.vkCmdDispatch = RightDispatch;
    right.vkCmdSetViewport = RightSetViewport;
    right.vkCmdSetScissor = RightSetScissor;
  }
  if (right.vkCmdBeginRendering) {
    right.vkCmdBeginRendering = RightBeginRendering;
  }
  right.vkCmdBindDescriptorSets = RightBindDescriptorSets;
  right.vkCmdBindIndexBuffer = RightBindIndexBuffer;
  right.vkCmdBindVertexBuffers = RightBindVertexBuffers;
  right.vkCmdClearColorImage = RightClearColorImage;
  right.vkCmdCopyBuffer = RightCopyBuffer;
  right.vkCmdCopyBufferToImage = RightCopyBufferToImage;
  right.vkCmdCopyImageToBuffer = RightCopyImageToBuffer;
  right.vkCmdCopyQueryPoolResults = RightCopyQueryPoolResults;
  right.vkCmdBeginQuery = RightBeginQuery;
  right.vkCmdEndQuery = RightEndQuery;
  right.vkCmdResetQueryPool = RightResetQueryPool;
  right.vkCmdWriteTimestamp = RightWriteTimestamp;
  right.vkCmdPipelineBarrier = RightPipelineBarrier;
  left = functions;
  if (Multiview()) {
    left.vkCmdPipelineBarrier = LeftPipelineBarrier;
    left.vkCmdBeginRenderPass = LeftBeginRenderPass;
    left.vkCmdBeginRendering = LeftBeginRendering;
    left.vkCmdEndRendering = LeftEndRendering;
    left.vkCmdBindDescriptorSets = LeftBindDescriptorSets;
    Log("stereo: multiview, passes into render targets draw both eyes; the rest runs per eye on twins");
  } else {
    Log("stereo: the right eye replays each submission on twin resources");
  }
}

const VulkanDevice::Functions& RightEyeFunctions() { return right; }

const VulkanDevice::Functions& LeftEyeFunctions() { return left; }

bool Multiview() {
  static const bool wanted = [] {
    const char* v = std::getenv("MCLA_VR_MULTIVIEW");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return wanted && Stereo();
}

void LayeredNextImage() {
  if (Multiview()) {
    layered_next = true;
  }
}

VkImageView ArrayViewOf(VkImageView view) {
  auto lock = ReadLock();
  const auto it = array_views.find(view);
  return it != array_views.end() ? it->second : VK_NULL_HANDLE;
}

uint32_t EyeLayer(VkImage image) {
  if (!replay_eye || !image) {
    return 0;
  }
  auto lock = ReadLock();
  return layered_images.count(image) ? uint32_t(replay_eye) : 0;
}

void SetMultiviewVariant(VkDescriptorSet set, VkDescriptorSet variant) {
  std::unique_lock lock{mutex};
  if (variant == VK_NULL_HANDLE) {
    multiview_variants.erase(set);  // the set's target is gone
  } else {
    multiview_variants[set] = variant;
  }
}

bool InMultiviewPass() { return multiview_pass; }

bool LastPassMultiview() { return last_pass_multiview; }

void BeginReplay() {
  if (++replay_generation == 0) {  // a new replay: the caches' entries are old
    replay_generation = 1;
  }
}

void SetReplayEye(int eye) { replay_eye = eye; }

void EndReplay() {
  replay_eye = 0;
  multiview_pass = right_in_pass = false;
}

void SetGpuThread() { gpu_thread = true; }

PFN_vkCmdPushDescriptorSetKHR push_descriptor_set_fn = nullptr;
uint32_t max_push_descriptors = 0;

void SetPushDescriptors(PFN_vkCmdPushDescriptorSetKHR push_descriptor_set, uint32_t max) {
  static const bool off = [] {  // MCLA_VR_PUSH=0: descriptor sets with twins, as before
    const char* v = std::getenv("MCLA_VR_PUSH");
    return v && v[0] == '0';
  }();
  if (off || !Stereo() || !push_descriptor_set) {
    return;
  }
  push_descriptor_set_fn = push_descriptor_set;
  max_push_descriptors = max;
  REXLOG_INFO("VR: pixel shader textures pushed per draw (up to {} bindings)", max);
}

bool PushTextures(size_t textures, size_t samplers) {
  const size_t descriptors = textures * (Multiview() ? 2 : 1) + samplers;
  return push_descriptor_set_fn && descriptors && descriptors <= max_push_descriptors;
}

VkImageView RightEyeView(VkImageView view) {
  if (!view) {
    return view;
  }
  auto lock = ReadLock();
  return EyeView(view);
}

void PushTextureDescriptors(VkCommandBuffer command_buffer, VkPipelineLayout layout, uint32_t set,
                            uint32_t texture_count, uint32_t sampler_count, const VkDescriptorImageInfo* infos) {
  if (RightSkipsGraphics()) {
    return;
  }
  VkWriteDescriptorSet writes[2];
  uint32_t count = 0;
  if (texture_count && Multiview()) {
    // Both eyes' views of each texture, in its binding's two elements (consecutive bindings: one write).
    thread_local std::vector<VkDescriptorImageInfo> pairs;
    pairs.resize(size_t(texture_count) * 2);
    for (uint32_t i = 0; i < texture_count; ++i) {
      pairs[2 * i] = infos[i];
      pairs[2 * i + 1] = infos[texture_count + sampler_count + i];
    }
    VkWriteDescriptorSet& write = writes[count++];
    write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstBinding = 0;
    write.descriptorCount = texture_count * 2;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = pairs.data();
  } else if (texture_count) {
    VkWriteDescriptorSet& write = writes[count++];
    write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstBinding = 0;
    write.descriptorCount = texture_count;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = replay_eye ? infos + texture_count + sampler_count : infos;
  }
  if (sampler_count) {
    VkWriteDescriptorSet& write = writes[count++];
    write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstBinding = texture_count;
    write.descriptorCount = sampler_count;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = infos + texture_count;
  }
  push_descriptor_set_fn(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, set, count, writes);
}

VkImage TwinOf(VkImage image) { return Find(image_twins, image); }

VkBuffer TwinBufferOf(VkBuffer buffer) { return Find(buffer_twins, buffer); }

std::atomic<double> video_memory_total_gb{0.0};

double VideoMemoryTotalGB() { return video_memory_total_gb.load(); }

double VideoMemoryFreeGB(const VulkanDevice* device) {
  const VulkanInstance::Functions& ifn = device->vulkan_instance()->functions();
  if (!ifn.vkGetPhysicalDeviceMemoryProperties2 || !device->extensions().ext_EXT_memory_budget) {
    return video_memory_total_gb.load();
  }
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  properties.pNext = &budget;
  ifn.vkGetPhysicalDeviceMemoryProperties2(device->physical_device(), &properties);
  uint64_t free_bytes = 0;
  for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
    if (properties.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      free_bytes = std::max(free_bytes, budget.heapBudget[i] > budget.heapUsage[i]
                                            ? uint64_t(budget.heapBudget[i] - budget.heapUsage[i])
                                            : uint64_t(0));
    }
  }
  return double(free_bytes) / double(1ull << 30);
}

double VideoMemoryNeededGB(uint32_t scale_area) {
  double gb_per_area = 1.05;
  if (const char* v = std::getenv("MCLA_VR_VRAM_GB_PER_AREA")) {
    gb_per_area = std::max(0.1, std::atof(v));
  }
  return 4.2 + gb_per_area * scale_area;
}

void FitDrawResolutionScale(const VulkanDevice* device, uint32_t& scale_x, uint32_t& scale_y) {
  const VulkanInstance::Functions& ifn = device->vulkan_instance()->functions();
  {
    VkPhysicalDeviceMemoryProperties memory_properties;
    ifn.vkGetPhysicalDeviceMemoryProperties(device->physical_device(), &memory_properties);
    uint64_t largest = 0;
    for (uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
      if (memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        largest = std::max(largest, uint64_t(memory_properties.memoryHeaps[i].size));
      }
    }
    video_memory_total_gb = double(largest) / double(1ull << 30);
  }
  if (!Stereo() || scale_x * scale_y <= 1) {
    return;
  }
  if (!ifn.vkGetPhysicalDeviceMemoryProperties2 || !device->extensions().ext_EXT_memory_budget) {
    return;
  }
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  properties.pNext = &budget;
  ifn.vkGetPhysicalDeviceMemoryProperties2(device->physical_device(), &properties);
  uint64_t free_bytes = 0;
  for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
    if (properties.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      const uint64_t heap_free =
          budget.heapBudget[i] > budget.heapUsage[i] ? budget.heapBudget[i] - budget.heapUsage[i] : 0;
      free_bytes = std::max(free_bytes, heap_free);
    }
  }
  // Measured (MCLA, 2026-10-27, sparse scaled-resolve buffer, idle render targets freed): peaks of 12.5 GB at 2x4
  // (area 8) and 22.9 GB at 3x6 (area 18): 4.2 GB + 1.05 GB per unit of area (VideoMemoryNeededGB); 2 GB kept back
  // for the compositor and the driver.
  const double gb_base = VideoMemoryNeededGB(0);
  const double gb_per_area = VideoMemoryNeededGB(1) - gb_base;
  const double free_gb = double(free_bytes) / double(1ull << 30);
  const uint32_t max_area = uint32_t(std::max(1.0, (free_gb - 2.0 - gb_base) / gb_per_area));
  const uint32_t old_x = scale_x, old_y = scale_y;
  if (scale_x * scale_y <= max_area) {
    REXLOG_INFO("VR: draw resolution scale {}x{} fits ({:.1f} GB free, about {:.1f} GB needed)", scale_x, scale_y,
                free_gb, gb_base + gb_per_area * scale_x * scale_y);
    return;
  }
  // Keep the shape where y is a whole multiple of x (the 1:2 eye steps); else shrink both.
  if (scale_y % scale_x == 0) {
    const uint32_t ratio = scale_y / scale_x;
    uint32_t x = scale_x;
    while (x > 1 && x * x * ratio > max_area) {
      --x;
    }
    scale_x = x;
    scale_y = std::max(1u, x * ratio);
    while (scale_y > 1 && scale_x * scale_y > max_area) {
      --scale_y;
    }
  } else {
    while (scale_x * scale_y > max_area && (scale_x > 1 || scale_y > 1)) {
      if (scale_y >= scale_x && scale_y > 1) {
        --scale_y;
      } else {
        --scale_x;
      }
    }
  }
  REXLOG_WARN("VR: draw resolution scale {}x{} needs about {:.1f} GB of video memory, {:.1f} GB is free: using {}x{}",
              old_x, old_y, gb_base + gb_per_area * old_x * old_y, free_gb, scale_x, scale_y);
}

void SparseTwinNext() { sparse_twin_next = true; }

uint64_t TwinBytes() { return twin_bytes.load(); }

void TwinSparseBinds(std::vector<VkSparseBufferMemoryBindInfo>& infos, std::vector<VkSparseMemoryBind>& binds) {
  if (!Stereo()) {
    return;
  }
  std::unique_lock lock{mutex};
  const size_t info_count = infos.size();
  // The twins' binds go after the left eye's: count them first so the pointers stay valid.
  size_t extra = 0;
  for (size_t i = 0; i < info_count; ++i) {
    if (buffer_twins.find(infos[i].buffer) != buffer_twins.end()) {
      extra += infos[i].bindCount;
    }
  }
  if (!extra) {
    return;
  }
  std::vector<VkSparseMemoryBind> twin_binds;
  twin_binds.reserve(extra);
  std::vector<std::pair<VkBuffer, std::pair<size_t, uint32_t>>> twin_infos;
  for (size_t i = 0; i < info_count; ++i) {
    const auto twin = buffer_twins.find(infos[i].buffer);
    if (twin == buffer_twins.end()) {
      continue;
    }
    const size_t first = twin_binds.size();
    for (uint32_t j = 0; j < infos[i].bindCount; ++j) {
      VkSparseMemoryBind bind = binds[uintptr_t(infos[i].pBinds) + j];  // pBinds: an offset into `binds` here
      if (bind.memory != VK_NULL_HANDLE) {
        const auto twin_memory = memory_twins.find(bind.memory);
        if (twin_memory == memory_twins.end()) {
          static bool logged = false;
          if (!logged) {
            logged = true;
            REXLOG_WARN("VR: sparse memory without a right-eye twin: the twin's pages stay unbound");
          }
          continue;
        }
        bind.memory = twin_memory->second;
      }
      twin_binds.push_back(bind);
    }
    twin_infos.push_back({twin->second, {first, uint32_t(twin_binds.size() - first)}});
  }
  const size_t base = binds.size();
  binds.insert(binds.end(), twin_binds.begin(), twin_binds.end());
  // The left eye's pointers into `binds` may have moved: the caller rebuilds them from offsets (see the caller).
  for (const auto& [buffer, range] : twin_infos) {
    VkSparseBufferMemoryBindInfo info;
    info.buffer = buffer;
    info.bindCount = range.second;
    info.pBinds = reinterpret_cast<const VkSparseMemoryBind*>(uintptr_t(base + range.first));  // an offset for now
    infos.push_back(info);
  }
}


VkImageView TwinViewOf(VkImageView view) { return Find(image_view_twins, view); }

bool IsViewEyeDependent(VkImageView view) {
  if (!view) {
    return false;
  }
  auto lock = ReadLock();
  return EyeView(view) != view;
}

bool AnyViewEyeDependent(const VkDescriptorImageInfo* infos, size_t count) {
  if (!count) {
    return false;
  }
  auto lock = ReadLock();
  for (size_t i = 0; i < count; ++i) {
    if (infos[i].imageView && EyeView(infos[i].imageView) != infos[i].imageView) {
      return true;
    }
  }
  return false;
}

void MarkTexture(VkImage image) {
  if (!Stereo()) {
    return;
  }
  std::unique_lock lock{mutex};
  if (const auto it = image_records.find(image); it != image_records.end()) {
    it->second.eye_independent = true;
  }
}

void MarkEyeDependent(VkImage image) {
  if (!Stereo()) {
    return;
  }
  std::unique_lock lock{mutex};
  if (const auto it = image_records.find(image); it != image_records.end()) {
    it->second.eye_independent = false;
  }
}

void BeginRightEye(VkCommandBuffer command_buffer) {
  // The eyes share the uploads and readbacks: the right eye starts after the left eye's commands.
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  real.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
  replay_eye = 1;
  if (++replay_generation == 0) {  // a new replay: the caches' entries are old
    replay_generation = 1;
  }
  static bool logged = false;
  if (!logged) {
    logged = true;
    REXLOG_INFO("VR: right-eye twins: {} MB of device memory, {} images, {} buffers, {} descriptor sets of {}",
                twin_bytes.load() >> 20, image_twins.size(), buffer_twins.size(), set_twins.size(), sets_made.load());
  }
}

void EndRightEye() {
  replay_eye = 0;
  if (counting) {
    static auto window = std::chrono::steady_clock::now();
    ++count_frames;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - window).count();
    if (seconds >= 2.0) {
      const double n = double(std::max<uint64_t>(1, count_frames.exchange(0)));
      REXLOG_INFO("VR: per submission (one eye): {:.0f} barriers, {:.0f} buffer copies ({:.0f} regions), {:.0f} "
                  "draws, {:.0f} dispatches, {:.0f} pipeline binds, {:.0f} descriptor binds, {:.0f} passes, {:.0f} "
                  "push constants, {:.0f} viewport/scissor",
                  count_barriers.exchange(0) / n, count_buffer_copies.exchange(0) / n,
                  count_copy_regions.exchange(0) / n, count_draws.exchange(0) / n, count_dispatches.exchange(0) / n,
                  count_pipelines.exchange(0) / n, count_set_binds.exchange(0) / n, count_passes.exchange(0) / n,
                  count_pushes.exchange(0) / n, count_dynamic_state.exchange(0) / n);
      for (int t = 0; t < kTimeCount; ++t) {
        static const char* names[] = {"allocate sets", "update sets", "  of which the driver", "other wrappers",
                                      "  of which the slow path"};
        REXLOG_INFO("VR: GPU thread wrappers: {} {:.0f} calls, {:.0f} kcycles a submission", names[t],
                    wrap_calls[t].exchange(0) / n, wrap_cycles[t].exchange(0) / n / 1000.0);
      }
      window = std::chrono::steady_clock::now();
    }
  }
}

int ReplayEye() { return replay_eye; }

const VkPushConstantRange& EyeShiftRange() {
  static const VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                         sizeof(EyePush)};
  return range;
}

void RecordEyeShift(VkCommandBuffer command_buffer, VkPipelineLayout layout, bool perspective,
                    const EyePush& push_template, const VulkanDevice::Functions& functions) {
  if (RightSkipsGraphics()) {
    return;
  }
  EyePush push = push_template;
  if (perspective) {  // multiview: both eyes' (the view picks)
    push.shift = EyeShift(Multiview() ? 0 : replay_eye);
    push.shift_right = Multiview() ? EyeShift(1) : push.shift;
  } else {
    push.shift = push.shift_right = push.jitter_x = push.jitter_y = push.motion = 0.f;
  }
  functions.vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(push), &push);
}

}  // namespace rex::ui::vulkan::vr
