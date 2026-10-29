// VR fork: OpenXR session, frame loop and the floating screen. See vr_openxr.h. Ported from the Uncharted port's
// renderer_vulkan/vr/vr_openxr.cpp (itself the Driveclub recomp's), with the same pacing.

#include <rex/ui/vulkan/vr_openxr.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/ui/keybinds.h>
#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/vr_dlss.h>
#include <rex/ui/vulkan/vr_twins.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

REXCVAR_DEFINE_BOOL(vr, false, "VR", "Play in a headset through OpenXR (or MCLA_VR=1)");
REXCVAR_DEFINE_INT32(vr_frame_cap, 1, "VR",
                     "Live: frames to OpenXR at 1/N of the headset's refresh (1 off, 2 half, 3 a third): the game "
                     "paced to it and the headset loop submitting at it, for an external frame generator");
REXCVAR_DEFINE_BOOL(vr_desktop_mirror, false, "VR",
                    "Show the game in its desktop window too while in VR (may stall the game now and then)");

namespace rex::ui::vulkan::vr {
namespace {

void Log(const char* format, ...) {
  char text[1024];
  va_list args;
  va_start(args, format);
  std::vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  REXLOG_INFO("VR: {}", text);
}

struct XrFunctions {
  PFN_xrGetInstanceProcAddr GetInstanceProcAddr{};
  PFN_xrGetDisplayRefreshRateFB GetDisplayRefreshRateFB{};
#define XR_FN(name) PFN_xr##name name{};
#define XR_FUNCTIONS                         \
  XR_FN(CreateInstance)                      \
  XR_FN(EnumerateInstanceExtensionProperties) \
  XR_FN(DestroyInstance)                     \
  XR_FN(GetSystem)                           \
  XR_FN(GetVulkanInstanceExtensionsKHR)      \
  XR_FN(GetVulkanDeviceExtensionsKHR)        \
  XR_FN(GetVulkanGraphicsDeviceKHR)          \
  XR_FN(GetVulkanGraphicsRequirementsKHR)    \
  XR_FN(CreateSession)                       \
  XR_FN(BeginSession)                        \
  XR_FN(EndSession)                          \
  XR_FN(PollEvent)                           \
  XR_FN(CreateReferenceSpace)                \
  XR_FN(EnumerateSwapchainFormats)           \
  XR_FN(CreateSwapchain)                     \
  XR_FN(DestroySwapchain)                    \
  XR_FN(EnumerateSwapchainImages)            \
  XR_FN(AcquireSwapchainImage)               \
  XR_FN(WaitSwapchainImage)                  \
  XR_FN(ReleaseSwapchainImage)               \
  XR_FN(WaitFrame)                           \
  XR_FN(BeginFrame)                          \
  XR_FN(EndFrame)                            \
  XR_FN(LocateViews)                         \
  XR_FN(LocateSpace)
  XR_FUNCTIONS
#undef XR_FN
};

// The Vulkan functions the layer uses, loaded from the device.
struct VkFunctions {
#define VK_FN(name) PFN_##name name{};
#define VK_FUNCTIONS                  \
  VK_FN(vkGetDeviceQueue)             \
  VK_FN(vkCreateCommandPool)          \
  VK_FN(vkAllocateCommandBuffers)     \
  VK_FN(vkCreateFence)                \
  VK_FN(vkWaitForFences)              \
  VK_FN(vkResetFences)                \
  VK_FN(vkResetCommandBuffer)         \
  VK_FN(vkBeginCommandBuffer)         \
  VK_FN(vkEndCommandBuffer)           \
  VK_FN(vkQueueSubmit)                \
  VK_FN(vkDeviceWaitIdle)             \
  VK_FN(vkCreateImage)                \
  VK_FN(vkDestroyImage)               \
  VK_FN(vkGetImageMemoryRequirements) \
  VK_FN(vkAllocateMemory)             \
  VK_FN(vkFreeMemory)                 \
  VK_FN(vkBindImageMemory)            \
  VK_FN(vkCmdPipelineBarrier)         \
  VK_FN(vkCmdBlitImage)               \
  VK_FN(vkCmdCopyImage)               \
  VK_FN(vkCreateSemaphore)            \
  VK_FN(vkGetSemaphoreCounterValue)   \
  VK_FN(vkWaitSemaphores)                \
  VK_FN(vkCreateImageView)               \
  VK_FN(vkDestroyImageView)              \
  VK_FN(vkCreateShaderModule)            \
  VK_FN(vkCreateDescriptorSetLayout)     \
  VK_FN(vkCreatePipelineLayout)          \
  VK_FN(vkCreateComputePipelines)        \
  VK_FN(vkCreateDescriptorPool)          \
  VK_FN(vkAllocateDescriptorSets)        \
  VK_FN(vkUpdateDescriptorSets)          \
  VK_FN(vkCmdBindPipeline)               \
  VK_FN(vkCmdBindDescriptorSets)         \
  VK_FN(vkCmdPushConstants)              \
  VK_FN(vkCmdDispatch)                   \
  VK_FN(vkCreateBuffer)                  \
  VK_FN(vkDestroyBuffer)                 \
  VK_FN(vkGetBufferMemoryRequirements)   \
  VK_FN(vkBindBufferMemory)              \
  VK_FN(vkMapMemory)                     \
  VK_FN(vkCmdCopyImageToBuffer)
  VK_FUNCTIONS
#undef VK_FN
  PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties{};
  PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties{};
};

struct State {
  std::once_flag init_once;
  XrFunctions xr;
  VkFunctions vk;
  uint32_t max_push_descriptors = 0;  // VK_KHR_push_descriptor enabled (stereo): its limit
  XrInstance instance{XR_NULL_HANDLE};
  XrSystemId system{XR_NULL_SYSTEM_ID};
  std::string instance_extension_text, device_extension_text;
  std::vector<const char*> instance_extensions, device_extensions;
  bool device_extensions_queried = false;

  const VulkanDevice* vulkan_device = nullptr;
  VkDevice device{};
  VkPhysicalDevice physical_device{};
  uint32_t queue_family = 0;
  XrSession session{XR_NULL_HANDLE};
  XrSpace local_space{XR_NULL_HANDLE};
  XrSpace view_space{XR_NULL_HANDLE};
  bool running = false;
  std::atomic<bool> recenter_requested{false};
  bool have_reference = false;
  float reference_yaw = 0.f;
  XrVector3f reference_position{};

  // The floating screen: an OpenXR swapchain (the headset's thread) and three images in its UNORM twin (the GPU
  // thread blits the frame into one; it is copied bit for bit into the sRGB swapchain image).
  XrSwapchain swapchain{XR_NULL_HANDLE};
  uint32_t width = 0, height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::vector<XrSwapchainImageVulkanKHR> images;
  static constexpr uint32_t kScreens = 6;  // the GPU runs 2-3 game frames behind: several in flight
  VkImage screen_images[kScreens]{};
  VkDeviceMemory screen_memory[kScreens]{};
  uint32_t screen_width = 0, screen_height = 0, screen_next = 0;

  // The HUD (a posed frame split at its first HUD draw, SplitFrame): an image per screen (the GPU thread copies the
  // HUD drawn over transparent black into it), its own swapchain, shown as a quad fixed in front of the seat.
  VkImage hud_screen_images[kScreens]{};
  VkDeviceMemory hud_screen_memory[kScreens]{};
  XrSwapchain hud_swapchain{XR_NULL_HANDLE};
  uint32_t hud_width = 0, hud_height = 0;
  std::atomic<float> hud_aspect{9.f / 16.f};  // the game's output (height over width, unscaled)
  std::vector<XrSwapchainImageVulkanKHR> hud_images;
  // The HUD's alpha for a compositor blending in linear light (vr_hud_alpha.comp), run by the headset's thread from
  // a HUD image (RGBA8, kept as the GPU thread copied it) into hud_fixed, copied into the swapchain bit for bit.
  VkImageView hud_screen_views[kScreens]{};
  VkImage hud_fixed{};
  VkDeviceMemory hud_fixed_memory{};
  VkImageView hud_fixed_view{};
  VkDescriptorSetLayout hud_set_layout{};
  VkPipelineLayout hud_pipeline_layout{};
  VkPipeline hud_pipeline{};
  VkDescriptorPool hud_pool{};
  VkDescriptorSet hud_sets[kScreens]{};
  int split_screen = -1;  // the GPU thread's: the screen of the frame being recorded, when split
  bool split_hud_captured = false;  // the GPU thread's: its HUD copy was recorded

  // The headset's frame thread and its own queue (a second queue of the graphics family when it has one).
  std::thread thread;
  std::atomic<bool> quit{false};
  uint32_t queue_index = 0;
  bool own_queue = false;
  VkQueue queue{};
  VkCommandPool pool{};
  VkCommandBuffer cmds[3]{};
  VkFence fences[3]{};
  uint32_t next_cmd = 0;
  std::mutex frame_mutex;

  // Publications: a screen image, ready on the GPU once `semaphore` (a timeline) reaches `tick`.
  struct Published {
    bool valid = false;
    bool posed = false;  // rendered with a head pose: shown as a projection layer at it, not on the screen
    bool hud = false;    // split: the HUD in hud_screen_images[screen], the scene alone in the screen
    float pose[16]{}, tangents[4]{};
    uint32_t screen = 0, width = 0, height = 0;
    uint32_t out_width = 0, out_height = 0;  // an eye in the headset's swapchain (dlss::OutputSize)
    uint64_t tick = 0;
    uint64_t serial = 0;
  };
  VkSemaphore timeline{};
  uint64_t timeline_value = 0;    // the GPU thread's
  Published pending;              // the GPU thread's: recorded, its submission not built yet
  std::mutex publish_mutex;
  std::array<Published, 5> recent{};  // newest first
  uint64_t published_serial = 0;
  std::condition_variable publish_cv;
  uint64_t shown_serial = 0;      // the headset's thread
  bool shown_posed = false;       // the headset's thread: the shown frame's pose (a projection layer at it)
  bool shown_hud = false;         // the headset's thread: the shown frame's HUD is in the HUD swapchain
  float shown_pose[16]{}, shown_tangents[4]{};

  // Pacing (Driveclub's): the headset's refresh period, the adaptive cap, a token per headset frame.
  std::atomic<uint64_t> display_period_ns{0};
  std::atomic<uint64_t> refresh_period_ns{0};
  std::atomic<bool> capped{false};
  // Slow frames (the game under 80% of the headset's rate, back at 95%, or the frame cap on): rendered this much wider
  // than the eyes' field of view, so the runtime has picture to turn into before the next frame (else a head turn
  // shows the frame's edge: the map transition "cropped on one side"). MCLA_VR_OVERSCAN sets the factor (1: none).
  std::atomic<bool> slow_frames{false};
  // BeginFrame: unposed frames since the last posed one, and whether the last frame shown was posed.
  uint32_t unposed_run = 0;
  bool last_shown_posed = false;
  std::atomic<bool> flip_token{false};
  std::mutex token_mutex;
  std::condition_variable token_cv;
  std::chrono::steady_clock::time_point flips_since{};
  uint32_t flips = 0;

  // Frame poses: the game thread's camera hook sets the frame's (SetFramePose), VdSwap queues it (OnGameSwap), the
  // GPU thread takes it at the swap packet (OnGpuSwap) for the frame's publication.
  std::mutex pose_mutex;
  bool frame_posed = false;
  bool frame_has_scene = true;  // the GPU thread's: the frame being finished drew a head-tracked 3D scene
  float frame_pose[16]{}, frame_tangents[4]{};
  struct SwapPose {
    bool posed;
    float pose[16], tangents[4];
    uint32_t frontbuffer = 0;  // the physical address its swap packet presents
  };
  std::deque<SwapPose> swap_poses;
  SwapPose gpu_pose{};  // the GPU thread's: the frame being presented
  // The pose the camera hook last put in the camera (update thread), and the one a frame's scene was drawn with (the
  // GPU thread, at its first head-tracked draw): SceneDrawStarted.
  std::mutex camera_mutex;
  bool camera_posed = false;
  float camera_pose[16]{}, camera_tangents[4]{};
  bool drawn_posed = false;
  float drawn_pose[16]{}, drawn_tangents[4]{};

  // The head for the game's next frame.
  std::mutex head_mutex;
  bool head_valid = false;
  float head_pose[16]{};
  float head_tangents[4]{};
  float ipd = 0.064f;
  std::atomic<float> eye_shift{0.f};  // the left eye's clip-space shift (EyeShift)

  // Frame statistics.
  std::vector<double> intervals;
  double last_submit = 0.0;
  std::vector<double> display_intervals;  // between the display times of submitted frames (a skipped one: two periods)
  int64_t last_display = 0;
  uint32_t repeats = 0;
  std::atomic<uint32_t> count_frames{0};

  XrFrameState frame_state{XR_TYPE_FRAME_STATE};
};

State g;

bool Check(XrResult result, const char* what) {
  if (XR_SUCCEEDED(result)) {
    return true;
  }
  Log("%s failed (%d)", what, int(result));
  return false;
}

// The runtime may submit on the queue it was given inside its calls: alone on its own queue, under the renderer's
// queue lock when it shares the renderer's.
struct QueueLock {
  std::optional<VulkanDevice::Queue::Acquisition> acquisition;
  QueueLock() {
    if (!g.own_queue && g.vulkan_device) {
      acquisition.emplace(g.vulkan_device->AcquireQueue(g.queue_family, 0));
    }
  }
};

bool LoadLoader() {
#ifdef _WIN32
  wchar_t exe[MAX_PATH]{};
  GetModuleFileNameW(nullptr, exe, MAX_PATH);
  std::wstring path(exe);
  path = path.substr(0, path.find_last_of(L"\\/") + 1) + L"openxr_loader.dll";
  HMODULE module = LoadLibraryW(path.c_str());
  if (!module) {
    module = LoadLibraryW(L"openxr_loader.dll");
  }
  if (!module) {
    return false;
  }
  g.xr.GetInstanceProcAddr =
      reinterpret_cast<PFN_xrGetInstanceProcAddr>(GetProcAddress(module, "xrGetInstanceProcAddr"));
#else
  void* module = dlopen("libopenxr_loader.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!module) {
    return false;
  }
  g.xr.GetInstanceProcAddr =
      reinterpret_cast<PFN_xrGetInstanceProcAddr>(dlsym(module, "xrGetInstanceProcAddr"));
#endif
  return g.xr.GetInstanceProcAddr != nullptr;
}

void LoadXrFunctions(XrInstance instance) {
#define XR_FN(name) \
  g.xr.GetInstanceProcAddr(instance, "xr" #name, reinterpret_cast<PFN_xrVoidFunction*>(&g.xr.name));
  XR_FUNCTIONS
#undef XR_FN
}

std::vector<const char*> Split(std::string& text) {
  std::vector<const char*> names;
  for (char& c : text) {
    if (c == ' ') {
      c = '\0';
    }
  }
  for (size_t i = 0; i < text.size();) {
    const size_t len = std::strlen(text.c_str() + i);
    if (len) {
      names.push_back(text.c_str() + i);
    }
    i += len + 1;
  }
  return names;
}

void Init() {
  if (const char* v = std::getenv("MCLA_VR"); !REXCVAR_GET(vr) && (!v || v[0] != '1')) {
    return;
  }
  if (!LoadLoader()) {
    Log("no openxr_loader.dll: VR off");
    return;
  }
  LoadXrFunctions(XR_NULL_HANDLE);
  bool refresh_rate = false;
  if (g.xr.EnumerateInstanceExtensionProperties) {
    uint32_t count = 0;
    g.xr.EnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr);
    std::vector<XrExtensionProperties> offered(count, {XR_TYPE_EXTENSION_PROPERTIES});
    g.xr.EnumerateInstanceExtensionProperties(nullptr, count, &count, offered.data());
    for (const auto& extension : offered) {
      refresh_rate |= std::strcmp(extension.extensionName, XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME) == 0;
    }
  }
  const char* extensions[] = {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME, XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME};
  XrInstanceCreateInfo info{XR_TYPE_INSTANCE_CREATE_INFO};
  std::strcpy(info.applicationInfo.applicationName, "Midnight Club LA VR");
  std::strcpy(info.applicationInfo.engineName, "mcla-recomp");
  info.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
  info.enabledExtensionCount = refresh_rate ? 2 : 1;
  info.enabledExtensionNames = extensions;
  if (!g.xr.CreateInstance || !Check(g.xr.CreateInstance(&info, &g.instance), "xrCreateInstance")) {
    g.instance = XR_NULL_HANDLE;
    return;
  }
  LoadXrFunctions(g.instance);
  if (refresh_rate) {
    g.xr.GetInstanceProcAddr(g.instance, "xrGetDisplayRefreshRateFB",
                             reinterpret_cast<PFN_xrVoidFunction*>(&g.xr.GetDisplayRefreshRateFB));
  }
  XrSystemGetInfo system_info{XR_TYPE_SYSTEM_GET_INFO};
  system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  if (!Check(g.xr.GetSystem(g.instance, &system_info, &g.system), "xrGetSystem (no headset?)")) {
    g.xr.DestroyInstance(g.instance);
    g.instance = XR_NULL_HANDLE;
    return;
  }
  uint32_t size = 0;
  g.xr.GetVulkanInstanceExtensionsKHR(g.instance, g.system, 0, &size, nullptr);
  g.instance_extension_text.resize(size);
  g.xr.GetVulkanInstanceExtensionsKHR(g.instance, g.system, size, &size, g.instance_extension_text.data());
  g.instance_extension_text.resize(size ? size - 1 : 0);
  g.instance_extensions = Split(g.instance_extension_text);
  dlss::AddInstanceExtensions(g.instance_extensions);  // DLSS's (vr_dlss), when wanted
  Log("OpenXR instance ready; Vulkan instance extensions: %s", g.instance_extension_text.c_str());
}

VkFormat UnormTwin(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8G8B8A8_SRGB:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB:
      return VK_FORMAT_B8G8R8A8_UNORM;
    default:
      return format;
  }
}

// Before freeing images the GPU may use: the renderer's queue (its pending submissions flushed by AcquireQueue) and
// the device idle.
void WaitIdle() {
  std::optional<VulkanDevice::Queue::Acquisition> acquisition;
  if (g.vulkan_device) {
    acquisition.emplace(g.vulkan_device->AcquireQueue(g.queue_family, 0));
  }
  g.vk.vkDeviceWaitIdle(g.device);
}

bool PickFormat() {
  uint32_t count = 0;
  g.xr.EnumerateSwapchainFormats(g.session, 0, &count, nullptr);
  std::vector<int64_t> formats(count);
  g.xr.EnumerateSwapchainFormats(g.session, count, &count, formats.data());
  g.format = VK_FORMAT_UNDEFINED;
  for (const VkFormat wanted : {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB}) {
    for (const int64_t f : formats) {
      if (g.format == VK_FORMAT_UNDEFINED && VkFormat(f) == wanted) {
        g.format = wanted;
      }
    }
  }
  if (g.format == VK_FORMAT_UNDEFINED) {
    Log("the runtime offers no 8-bit sRGB swapchain format");
    return false;
  }
  return true;
}

void DestroySwapchain() {
  if (g.swapchain) {
    WaitIdle();
    QueueLock lock;
    g.xr.DestroySwapchain(g.swapchain);
    g.swapchain = XR_NULL_HANDLE;
  }
  g.images.clear();
}

bool CreateHudSwapchain(uint32_t width, uint32_t height) {
  if (g.hud_swapchain) {
    WaitIdle();
    QueueLock lock;
    g.xr.DestroySwapchain(g.hud_swapchain);
    g.hud_swapchain = XR_NULL_HANDLE;
  }
  XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
  info.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
  info.format = g.format;
  info.sampleCount = 1;
  info.width = width;
  info.height = height;
  info.faceCount = 1;
  info.arraySize = 1;
  info.mipCount = 1;
  {
    QueueLock lock;
    if (!Check(g.xr.CreateSwapchain(g.session, &info, &g.hud_swapchain), "xrCreateSwapchain (HUD)")) {
      return false;
    }
  }
  uint32_t count = 0;
  g.xr.EnumerateSwapchainImages(g.hud_swapchain, 0, &count, nullptr);
  g.hud_images.assign(count, XrSwapchainImageVulkanKHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
  g.xr.EnumerateSwapchainImages(g.hud_swapchain, count, &count,
                                reinterpret_cast<XrSwapchainImageBaseHeader*>(g.hud_images.data()));
  g.hud_width = width;
  g.hud_height = height;
  Log("HUD swapchain %ux%u", width, height);
  return true;
}

bool CreateSwapchain(uint32_t width, uint32_t height) {
  DestroySwapchain();
  XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
  info.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
  info.format = g.format;
  info.sampleCount = 1;
  info.width = width;
  info.height = height;
  info.faceCount = 1;
  info.arraySize = 1;
  info.mipCount = 1;
  {
    QueueLock lock;
    if (!Check(g.xr.CreateSwapchain(g.session, &info, &g.swapchain), "xrCreateSwapchain")) {
      return false;
    }
  }
  uint32_t count = 0;
  g.xr.EnumerateSwapchainImages(g.swapchain, 0, &count, nullptr);
  g.images.assign(count, XrSwapchainImageVulkanKHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
  g.xr.EnumerateSwapchainImages(g.swapchain, count, &count,
                                reinterpret_cast<XrSwapchainImageBaseHeader*>(g.images.data()));
  g.width = width;
  g.height = height;
  Log("screen swapchain %ux%u, VkFormat %d, %u images", width, height, int(g.format), count);
  return true;
}

// An image in device memory for transfers (the UNORM twin of the swapchain format).
bool MakeTransferImage(uint32_t width, uint32_t height, VkImage& image, VkDeviceMemory& memory,
                       VkFormat format = VK_FORMAT_UNDEFINED, VkImageUsageFlags usage = 0) {
  VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format != VK_FORMAT_UNDEFINED ? format : UnormTwin(g.format);
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | usage;
  if (g.vk.vkCreateImage(g.device, &image_info, nullptr, &image) != VK_SUCCESS) {
    image = {};
    return false;
  }
  VkPhysicalDeviceMemoryProperties properties{};
  g.vk.vkGetPhysicalDeviceMemoryProperties(g.physical_device, &properties);
  VkMemoryRequirements requirements{};
  g.vk.vkGetImageMemoryRequirements(g.device, image, &requirements);
  uint32_t type = 0;
  for (; type < properties.memoryTypeCount; ++type) {
    if ((requirements.memoryTypeBits & (1u << type)) &&
        (properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      break;
    }
  }
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = requirements.size;
  alloc.memoryTypeIndex = type;
  if (g.vk.vkAllocateMemory(g.device, &alloc, nullptr, &memory) != VK_SUCCESS) {
    return false;
  }
  g.vk.vkBindImageMemory(g.device, image, memory, 0);
  return true;
}

VkImageView MakeView(VkImage image, VkFormat format) {
  VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  info.image = image;
  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.format = format;
  info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view{};
  g.vk.vkCreateImageView(g.device, &info, nullptr, &view);
  return view;
}

// The HUD alpha pass's pipeline (once).
bool MakeHudPipeline() {
  if (g.hud_pipeline) {
    return true;
  }
  static const uint32_t code[] = {
#include "shaders/vr_hud_alpha.inc"
  };
  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = sizeof(code);
  module_info.pCode = code;
  VkShaderModule module{};
  if (g.vk.vkCreateShaderModule(g.device, &module_info, nullptr, &module) != VK_SUCCESS) {
    return false;
  }
  VkDescriptorSetLayoutBinding bindings[2]{};
  for (uint32_t i = 0; i < 2; ++i) {
    bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 2;
  set_info.pBindings = bindings;
  g.vk.vkCreateDescriptorSetLayout(g.device, &set_info, nullptr, &g.hud_set_layout);
  const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &g.hud_set_layout;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &range;
  g.vk.vkCreatePipelineLayout(g.device, &layout_info, nullptr, &g.hud_pipeline_layout);
  VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = module;
  pipeline_info.stage.pName = "main";
  pipeline_info.layout = g.hud_pipeline_layout;
  if (g.vk.vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &g.hud_pipeline) !=
      VK_SUCCESS) {
    return false;
  }
  const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * State::kScreens};
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = State::kScreens;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &size;
  g.vk.vkCreateDescriptorPool(g.device, &pool_info, nullptr, &g.hud_pool);
  VkDescriptorSetLayout layouts[State::kScreens];
  std::fill(std::begin(layouts), std::end(layouts), g.hud_set_layout);
  VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate.descriptorPool = g.hud_pool;
  allocate.descriptorSetCount = State::kScreens;
  allocate.pSetLayouts = layouts;
  return g.vk.vkAllocateDescriptorSets(g.device, &allocate, g.hud_sets) == VK_SUCCESS;
}

// The GPU thread: the screen images at the frame's size (the UNORM twin of the swapchain format), and the HUD's.
bool MakeScreens(uint32_t width, uint32_t height, uint32_t hud_width, uint32_t hud_height) {
  if (g.screen_images[0] && g.screen_width == width && g.screen_height == height) {
    return true;
  }
  std::unique_lock frame_lock{g.frame_mutex, std::defer_lock};
  if (g.screen_images[0]) {
    frame_lock.lock();
    WaitIdle();
    for (uint32_t i = 0; i < State::kScreens; ++i) {
      g.vk.vkDestroyImage(g.device, g.screen_images[i], nullptr);
      g.vk.vkFreeMemory(g.device, g.screen_memory[i], nullptr);
      g.vk.vkDestroyImageView(g.device, g.hud_screen_views[i], nullptr);
      g.vk.vkDestroyImage(g.device, g.hud_screen_images[i], nullptr);
      g.vk.vkFreeMemory(g.device, g.hud_screen_memory[i], nullptr);
      g.hud_screen_views[i] = {};
      g.screen_images[i] = {};
      g.screen_memory[i] = {};
      g.hud_screen_images[i] = {};
      g.hud_screen_memory[i] = {};
    }
  }
  if (g.hud_fixed) {
    g.vk.vkDestroyImageView(g.device, g.hud_fixed_view, nullptr);
    g.vk.vkDestroyImage(g.device, g.hud_fixed, nullptr);
    g.vk.vkFreeMemory(g.device, g.hud_fixed_memory, nullptr);
    g.hud_fixed = {};
  }
  if (!MakeHudPipeline() ||
      !MakeTransferImage(hud_width, hud_height, g.hud_fixed, g.hud_fixed_memory, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_STORAGE_BIT)) {
    return false;
  }
  g.hud_fixed_view = MakeView(g.hud_fixed, VK_FORMAT_R8G8B8A8_UNORM);
  for (uint32_t i = 0; i < State::kScreens; ++i) {
    if (!MakeTransferImage(width, height, g.screen_images[i], g.screen_memory[i]) ||
        !MakeTransferImage(hud_width, hud_height, g.hud_screen_images[i], g.hud_screen_memory[i],
                           VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT)) {
      return false;
    }
    g.hud_screen_views[i] = MakeView(g.hud_screen_images[i], VK_FORMAT_R8G8B8A8_UNORM);
    VkDescriptorImageInfo images[2] = {{VK_NULL_HANDLE, g.hud_screen_views[i], VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, g.hud_fixed_view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = g.hud_sets[i];
    write.descriptorCount = 2;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = images;
    g.vk.vkUpdateDescriptorSets(g.device, 1, &write, 0, nullptr);
  }
  g.screen_width = width;
  g.screen_height = height;
  return true;
}

void ReadRefreshRate() {
  float hz = 0.f;
  if (g.xr.GetDisplayRefreshRateFB && g.session &&
      XR_SUCCEEDED(g.xr.GetDisplayRefreshRateFB(g.session, &hz)) && hz > 10.f) {
    g.refresh_period_ns.store(uint64_t(1e9 / double(hz)), std::memory_order_relaxed);
    Log("headset refresh rate %.1f Hz", hz);
  }
}

void PollEvents() {
  XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
  while (g.xr.PollEvent(g.instance, &event) == XR_SUCCESS) {
    if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
      const auto& changed = reinterpret_cast<const XrEventDataSessionStateChanged&>(event);
      Log("session state %d", int(changed.state));
      if (changed.state == XR_SESSION_STATE_READY) {
        XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
        begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        g.running = Check(g.xr.BeginSession(g.session, &begin), "xrBeginSession");
        ReadRefreshRate();
      } else if (changed.state == XR_SESSION_STATE_STOPPING) {
        g.xr.EndSession(g.session);
        g.running = false;
      }
    } else if (event.type == XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB) {
      ReadRefreshRate();
    } else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
      g.recenter_requested = true;
    }
    event = {XR_TYPE_EVENT_DATA_BUFFER};
  }
}

void Barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
             VkAccessFlags dst_access) {
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = from;
  barrier.newLayout = to;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, EyeLayer(image), 1};  // (multiview: the eye's layer)
  g.vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                            nullptr, 0, nullptr, 1, &barrier);
}

// The head in the seat's space at the predicted display time, and the eyes' fields of view, for the camera.
// VR fork, development (MCLA_VR_POSE_TRACE=1): the head yaw (from a pose's z row) at each stage, when it changes.
std::atomic<uint32_t> g_vr_gpu_swaps{0};  // development: the pose trace's frame numbers

void PoseTrace(const char* stage, const float* pose) {
  static const bool on = std::getenv("MCLA_VR_POSE_TRACE") && std::getenv("MCLA_VR_POSE_TRACE")[0] == '1';
  if (!on) {
    return;
  }
  static std::mutex mutex;
  static std::map<std::string, float> last;
  const float yaw = std::atan2(pose[8], pose[10]) * 57.2958f;
  std::scoped_lock lock{mutex};
  auto& l = last[stage];
  if (std::fabs(l - yaw) > 0.05f) {
    l = yaw;
    Log("pose trace %s yaw %.2f (GPU swaps so far %u)", stage, yaw, g_vr_gpu_swaps.load());
  }
}

void UpdateHead() {
  XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
  if (!XR_SUCCEEDED(g.xr.LocateSpace(g.view_space, g.local_space, g.frame_state.predictedDisplayTime, &location)) ||
      !(location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
    return;
  }
  XrViewLocateInfo info{XR_TYPE_VIEW_LOCATE_INFO};
  info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  info.displayTime = g.frame_state.predictedDisplayTime;
  info.space = g.local_space;
  XrViewState state{XR_TYPE_VIEW_STATE};
  XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
  uint32_t count = 0;
  if (!XR_SUCCEEDED(g.xr.LocateViews(g.session, &info, &state, 2, &count, views)) || count != 2) {
    return;
  }
  XrQuaternionf q = location.pose.orientation;
  XrVector3f p = location.pose.position;
  const auto yaw_of = [](const XrQuaternionf& o) {
    const float fx = -2.f * (o.x * o.z + o.w * o.y), fz = -(1.f - 2.f * (o.x * o.x + o.y * o.y));
    return std::atan2(-fx, -fz);
  };
  if (!g.have_reference || g.recenter_requested) {
    g.reference_yaw = yaw_of(q);
    g.reference_position = p;
    g.have_reference = true;
    g.recenter_requested = false;
    Log("recentred (yaw %.1f degrees)", g.reference_yaw * 57.2958f);
  }
  {
    const float h = -g.reference_yaw * 0.5f;
    const XrQuaternionf r{0.f, std::sin(h), 0.f, std::cos(h)};
    const XrQuaternionf o = q;
    q = {r.w * o.x + r.x * o.w + r.y * o.z - r.z * o.y, r.w * o.y - r.x * o.z + r.y * o.w + r.z * o.x,
         r.w * o.z + r.x * o.y - r.y * o.x + r.z * o.w, r.w * o.w - r.x * o.x - r.y * o.y - r.z * o.z};
    const float dx = p.x - g.reference_position.x, dy = p.y - g.reference_position.y,
                dz = p.z - g.reference_position.z;
    const float c = std::cos(-g.reference_yaw), s = std::sin(-g.reference_yaw);
    p = {c * dx + s * dz, dy, -s * dx + c * dz};
  }
  const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z, xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z,
              wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
  const float pose[16] = {1 - 2 * (yy + zz), 2 * (xy + wz),     2 * (xz - wy),     0,
                          2 * (xy - wz),     1 - 2 * (xx + zz), 2 * (yz + wx),     0,
                          2 * (xz + wy),     2 * (yz - wx),     1 - 2 * (xx + yy), 0,
                          p.x,               p.y,               p.z,               1};
  // No overscan: rendering a wider field for head turns at low frame rates (1.15x, slow frames or a frame cap)
  // showed the middle 87% of the eye's pixels on the panel, 1.15x coarser in each direction at every size ("the source
  // pixels are bigger than they should be", Matt 2026-10-28). MCLA_VR_OVERSCAN=1.15 for the old behaviour.
  static const float overscan_factor = [] {
    const char* v = std::getenv("MCLA_VR_OVERSCAN");
    return v ? std::max(1.f, float(std::atof(v))) : 1.f;
  }();
  const float overscan =
      g.slow_frames.load(std::memory_order_relaxed) || REXCVAR_GET(vr_frame_cap) > 1 ? overscan_factor : 1.f;
  const float tangents[4] = {overscan * std::tan(std::min(views[0].fov.angleLeft, views[1].fov.angleLeft)),
                             overscan * std::tan(std::max(views[0].fov.angleRight, views[1].fov.angleRight)),
                             overscan * std::tan(std::max(views[0].fov.angleUp, views[1].fov.angleUp)),
                             overscan * std::tan(std::min(views[0].fov.angleDown, views[1].fov.angleDown))};
  const float ex = views[1].pose.position.x - views[0].pose.position.x,
              ey = views[1].pose.position.y - views[0].pose.position.y,
              ez = views[1].pose.position.z - views[0].pose.position.z;
  if (!g.head_valid) {
    Log("eye fields of view (tangents): left %.3f %.3f %.3f %.3f, right %.3f %.3f %.3f %.3f",
        std::tan(views[0].fov.angleLeft), std::tan(views[0].fov.angleRight), std::tan(views[0].fov.angleUp),
        std::tan(views[0].fov.angleDown), std::tan(views[1].fov.angleLeft), std::tan(views[1].fov.angleRight),
        std::tan(views[1].fov.angleUp), std::tan(views[1].fov.angleDown));
  }
  PoseTrace("0 UpdateHead", pose);
  std::scoped_lock lock{g.head_mutex};
  std::memcpy(g.head_pose, pose, sizeof(pose));
  std::memcpy(g.head_tangents, tangents, sizeof(tangents));
  g.head_valid = true;
  g.ipd = std::sqrt(ex * ex + ey * ey + ez * ez);
  // The camera's frustum is the eyes' union made symmetric (the game's camera hook): x' = x + P00 * baseline / 2.
  g.eye_shift.store(0.5f * StereoBaseline() / std::max(-tangents[0], tangents[1]), std::memory_order_relaxed);
}

// Every 900 submitted headset frames: rates, 1% low, late frames, repeats.
void NoteFrameTime() {
  const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (g.last_submit > 0.0) {
    g.intervals.push_back(t - g.last_submit);
  }
  g.last_submit = t;
  if (g.last_display) {
    g.display_intervals.push_back(double(g.frame_state.predictedDisplayTime - g.last_display) * 1e-9);
  }
  g.last_display = g.frame_state.predictedDisplayTime;
  if (g.intervals.size() < 900) {
    return;
  }
  const double period =
      g.frame_state.predictedDisplayPeriod > 0 ? double(g.frame_state.predictedDisplayPeriod) * 1e-9 : 1.0 / 90.0;
  double sum = 0.0;
  uint32_t late = 0, over = 0;
  for (const double i : g.intervals) {
    sum += i;
    late += i > 1.5 * period;
    over += i > 1.1 * period;
  }
  std::sort(g.intervals.begin(), g.intervals.end());
  const double p99 = g.intervals[g.intervals.size() * 99 / 100];
  // What the headset shows: display slots without a submitted frame (the runtime repeats one).
  uint32_t skipped = 0;
  double display_sum = 0.0;
  for (const double i : g.display_intervals) {
    display_sum += i;
    skipped += uint32_t(std::max(0.0, std::round(i / period) - 1.0));
  }
  std::sort(g.display_intervals.begin(), g.display_intervals.end());
  const double display_p99 =
      g.display_intervals.empty() ? period : g.display_intervals[g.display_intervals.size() * 99 / 100];
  const double new_frames = double(g.intervals.size() - std::min<size_t>(g.repeats, g.intervals.size()));
  {
    uint64_t value = 0;
    g.vk.vkGetSemaphoreCounterValue(g.device, g.timeline, &value);
    std::scoped_lock lock{g.publish_mutex};
    Log("timeline at %llu, newest publication %llu (serial %llu), shown serial %llu", (unsigned long long)value,
        (unsigned long long)g.recent[0].tick, (unsigned long long)g.recent[0].serial,
        (unsigned long long)g.shown_serial);
  }
  Log("game %.1f FPS (new frames), recorded %.1f; headset %.1f FPS, 1%% low %.1f FPS, over a period %.2f%%, late "
      "%.2f%% (display %.1f Hz)%s; displayed: 1%% low %.1f FPS, %u of %zu slots skipped (submit jitter is the "
      "first 1%% low)",
      new_frames / sum, g.count_frames.exchange(0) / sum, double(g.intervals.size()) / sum, 1.0 / p99,
      100.0 * over / double(g.intervals.size()), 100.0 * late / double(g.intervals.size()), 1.0 / period,
      g.capped ? ", capped" : "", 1.0 / display_p99, skipped, g.display_intervals.size() + skipped);
  g.intervals.clear();
  g.display_intervals.clear();
  g.repeats = 0;
}

VkCommandBuffer NextCommand() {
  const uint32_t i = g.next_cmd;
  g.next_cmd = (i + 1) % 3;
  g.vk.vkWaitForFences(g.device, 1, &g.fences[i], VK_TRUE, UINT64_MAX);
  g.vk.vkResetFences(g.device, 1, &g.fences[i]);
  g.vk.vkResetCommandBuffer(g.cmds[i], 0);
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  g.vk.vkBeginCommandBuffer(g.cmds[i], &begin);
  return g.cmds[i];
}

// VR fork, development: MCLA_VR_EYE_SHOT=<path>.bmp: when <path>.bmp.req appears, the next picture sent to the
// headset (both eyes side by side, at full size) is saved as a 32-bit BMP and the request file removed.
struct EyeShot {
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  uint32_t width = 0, height = 0;
};

EyeShot StartEyeShot(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height) {
  EyeShot shot;
  static const char* path = std::getenv("MCLA_VR_EYE_SHOT");
  if (!path || !std::filesystem::exists(std::string(path) + ".req")) {
    return shot;
  }
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = VkDeviceSize(width) * height * 4;
  info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (g.vk.vkCreateBuffer(g.device, &info, nullptr, &shot.buffer) != VK_SUCCESS) {
    return {};
  }
  VkMemoryRequirements requirements{};
  g.vk.vkGetBufferMemoryRequirements(g.device, shot.buffer, &requirements);
  VkPhysicalDeviceMemoryProperties properties{};
  g.vk.vkGetPhysicalDeviceMemoryProperties(g.physical_device, &properties);
  const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t type = 0;
  while (type < properties.memoryTypeCount && !((requirements.memoryTypeBits & (1u << type)) &&
                                                (properties.memoryTypes[type].propertyFlags & wanted) == wanted)) {
    ++type;
  }
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = requirements.size;
  alloc.memoryTypeIndex = type;
  if (type == properties.memoryTypeCount ||
      g.vk.vkAllocateMemory(g.device, &alloc, nullptr, &shot.memory) != VK_SUCCESS) {
    g.vk.vkDestroyBuffer(g.device, shot.buffer, nullptr);
    return {};
  }
  g.vk.vkBindBufferMemory(g.device, shot.buffer, shot.memory, 0);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {width, height, 1};
  g.vk.vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shot.buffer, 1, &region);
  shot.width = width;
  shot.height = height;
  return shot;
}

void FinishEyeShot(EyeShot& shot, VkFence fence) {
  if (!shot.buffer) {
    return;
  }
  static const char* path = std::getenv("MCLA_VR_EYE_SHOT");
  g.vk.vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX);
  void* data = nullptr;
  if (g.vk.vkMapMemory(g.device, shot.memory, 0, VK_WHOLE_SIZE, 0, &data) == VK_SUCCESS) {
    const bool rgba = UnormTwin(g.format) != VK_FORMAT_B8G8R8A8_UNORM;
    const uint32_t row = shot.width * 4, size = row * shot.height;
    uint8_t header[54] = {'B', 'M'};
    auto put = [&](int at, uint32_t v) { std::memcpy(header + at, &v, 4); };
    put(2, 54 + size);
    put(10, 54);
    put(14, 40);
    put(18, shot.width);
    put(22, uint32_t(-int32_t(shot.height)));  // top-down
    header[26] = 1;
    header[28] = 32;
    put(34, size);
    if (FILE* f = std::fopen(path, "wb")) {
      std::fwrite(header, 1, sizeof(header), f);
      std::vector<uint8_t> line(row);
      for (uint32_t y = 0; y < shot.height; ++y) {
        std::memcpy(line.data(), static_cast<const uint8_t*>(data) + size_t(y) * row, row);
        if (rgba) {
          for (uint32_t x = 0; x < row; x += 4) {
            std::swap(line[x], line[x + 2]);
          }
        }
        std::fwrite(line.data(), 1, row, f);
      }
      std::fclose(f);
      Log("eye shot %ux%u saved", shot.width, shot.height);
    }
  }
  g.vk.vkFreeMemory(g.device, shot.memory, nullptr);  // (unmapped with it)
  g.vk.vkDestroyBuffer(g.device, shot.buffer, nullptr);
  std::error_code ec;
  std::filesystem::remove(std::string(path) + ".req", ec);
  shot = {};
}

void SubmitCommand(VkCommandBuffer cmd, const State::Published& p) {
  g.vk.vkEndCommandBuffer(cmd);
  const uint32_t i = (g.next_cmd + 2) % 3;
  VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = 1;
  timeline.pWaitSemaphoreValues = &p.tick;
  const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.pNext = &timeline;
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &g.timeline;
  submit.pWaitDstStageMask = &stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  QueueLock lock;
  g.vk.vkQueueSubmit(g.queue, 1, &submit, g.fences[i]);
}

// The headset's thread: a new publication into the screen's swapchain.
bool Show(const State::Published& p) {
  const uint32_t width = p.out_width * (Stereo() ? 2 : 1);  // stereo: the eyes side by side
  if (width != g.screen_width || p.out_height != g.screen_height) {
    return false;  // VR fork: published before the screens changed size (a new headset size): its screen is gone
  }
  if ((g.width != width || g.height != p.out_height || !g.swapchain) && !CreateSwapchain(width, p.out_height)) {
    return false;
  }
  uint32_t index = 0;
  {
    QueueLock lock;
    XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (!Check(g.xr.AcquireSwapchainImage(g.swapchain, &acquire, &index), "xrAcquireSwapchainImage")) {
      return false;
    }
    XrSwapchainImageWaitInfo image_wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    image_wait.timeout = XR_INFINITE_DURATION;
    g.xr.WaitSwapchainImage(g.swapchain, &image_wait);
  }
  const VkCommandBuffer cmd = NextCommand();
  const VkImage target = g.images[index].image;
  Barrier(cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
          VK_ACCESS_TRANSFER_WRITE_BIT);
  VkImageCopy copy{};
  copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.extent = {g.width, g.height, 1};
  g.vk.vkCmdCopyImage(cmd, g.screen_images[p.screen], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  Barrier(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
  EyeShot shot = StartEyeShot(cmd, g.screen_images[p.screen], g.width, g.height);  // VR fork, development
  if (p.hud && (g.hud_width == p.width && g.hud_height == p.height ? true : CreateHudSwapchain(p.width, p.height))) {
    uint32_t hud_index = 0;
    {
      QueueLock lock;
      XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
      g.xr.AcquireSwapchainImage(g.hud_swapchain, &acquire, &hud_index);
      XrSwapchainImageWaitInfo image_wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
      image_wait.timeout = XR_INFINITE_DURATION;
      g.xr.WaitSwapchainImage(g.hud_swapchain, &image_wait);
    }
    const VkImage hud_target = g.hud_images[hud_index].image;
    // The alpha for linear-light blending (and the channel order), into hud_fixed.
    Barrier(cmd, g.hud_screen_images[p.screen], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    Barrier(cmd, g.hud_fixed, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT);
    g.vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.hud_pipeline);
    g.vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.hud_pipeline_layout, 0, 1,
                                 &g.hud_sets[p.screen], 0, nullptr);
    const uint32_t constants[3] = {p.width, p.height,
                                   uint32_t(UnormTwin(g.format) == VK_FORMAT_B8G8R8A8_UNORM)};
    g.vk.vkCmdPushConstants(cmd, g.hud_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);
    g.vk.vkCmdDispatch(cmd, (p.width + 7) / 8, (p.height + 7) / 8, 1);
    Barrier(cmd, g.hud_fixed, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    Barrier(cmd, g.hud_screen_images[p.screen], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    Barrier(cmd, hud_target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy hud_copy{};
    hud_copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    hud_copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    hud_copy.extent = {p.width, p.height, 1};
    g.vk.vkCmdCopyImage(cmd, g.hud_fixed, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, hud_target,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &hud_copy);
    Barrier(cmd, hud_target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
    g.shown_hud = true;
  } else {
    g.shown_hud = false;
  }
  SubmitCommand(cmd, p);
  FinishEyeShot(shot, g.fences[(g.next_cmd + 2) % 3]);
  g.shown_posed = p.posed;
  std::memcpy(g.shown_pose, p.pose, sizeof(g.shown_pose));
  PoseTrace("5 headset Show", g.shown_pose);
  std::memcpy(g.shown_tangents, p.tangents, sizeof(g.shown_tangents));
  {
    XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    QueueLock lock;
    g.xr.ReleaseSwapchainImage(g.swapchain, &release);
    if (g.shown_hud) {
      g.xr.ReleaseSwapchainImage(g.hud_swapchain, &release);
    }
  }
  return true;
}

void EndFrame() {
  // The screen: 3 m wide, 2.5 m ahead of the seat, at eye height, the picture's aspect.
  XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
  quad.space = g.local_space;
  quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
  quad.subImage.swapchain = g.swapchain;
  const uint32_t eye_width = g.width / (Stereo() ? 2 : 1);
  quad.subImage.imageRect = {{0, 0}, {int32_t(eye_width), int32_t(g.height)}};
  quad.pose.orientation.w = 1.f;
  quad.pose.position = {0.f, 0.f, -2.5f};
  // The game's own aspect: the picture's pixels are not square (the eyes are drawn taller than the output).
  quad.size = {3.f, 3.f * g.hud_aspect.load(std::memory_order_relaxed)};
  const XrCompositionLayerBaseHeader* layers[1] = {reinterpret_cast<XrCompositionLayerBaseHeader*>(&quad)};
  // A frame rendered with a head pose: a projection layer at that pose (the runtime turns it to the head's pose now),
  // the same picture in both eyes (no stereo yet).
  XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  XrCompositionLayerProjectionView views[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                               {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
  if (g.shown_posed && g.swapchain) {
    // The pose rows: the head's x, y, z axes and position (seat space, which is LOCAL turned and moved to the
    // reference). Back to LOCAL: rotate by the reference yaw and add its position.
    const float* m = g.shown_pose;
    const float yaw = g.reference_yaw, c = std::cos(yaw), s = std::sin(yaw);
    // The rotation matrix to a quaternion (rows are the axes' images).
    const float t = m[0] + m[5] + m[10];
    XrQuaternionf q{};
    if (t > 0.f) {
      const float r = std::sqrt(1.f + t) * 2.f;
      q = {(m[6] - m[9]) / r, (m[8] - m[2]) / r, (m[1] - m[4]) / r, 0.25f * r};
    } else if (m[0] > m[5] && m[0] > m[10]) {
      const float r = std::sqrt(1.f + m[0] - m[5] - m[10]) * 2.f;
      q = {0.25f * r, (m[1] + m[4]) / r, (m[8] + m[2]) / r, (m[6] - m[9]) / r};
    } else if (m[5] > m[10]) {
      const float r = std::sqrt(1.f + m[5] - m[0] - m[10]) * 2.f;
      q = {(m[1] + m[4]) / r, 0.25f * r, (m[6] + m[9]) / r, (m[8] - m[2]) / r};
    } else {
      const float r = std::sqrt(1.f + m[10] - m[0] - m[5]) * 2.f;
      q = {(m[8] + m[2]) / r, (m[6] + m[9]) / r, 0.25f * r, (m[1] - m[4]) / r};
    }
    const XrQuaternionf ry{0.f, std::sin(yaw * 0.5f), 0.f, std::cos(yaw * 0.5f)};
    const XrQuaternionf world{ry.w * q.x + ry.x * q.w + ry.y * q.z - ry.z * q.y,
                              ry.w * q.y - ry.x * q.z + ry.y * q.w + ry.z * q.x,
                              ry.w * q.z + ry.x * q.y - ry.y * q.x + ry.z * q.w,
                              ry.w * q.w - ry.x * q.x - ry.y * q.y - ry.z * q.z};
    const XrVector3f position{g.reference_position.x + c * m[12] + s * m[14], g.reference_position.y + m[13],
                              g.reference_position.z - s * m[12] + c * m[14]};
    // Stereo: the eyes half the IPD along the head's x axis (in LOCAL), each on its half of the image.
    const float half_ipd = Stereo() ? 0.5f * g.ipd : 0.f;
    const XrVector3f right{c * m[0] + s * m[2], m[1], -s * m[0] + c * m[2]};
    for (int eye = 0; eye < 2; ++eye) {
      const float side = eye ? half_ipd : -half_ipd;
      views[eye].pose.orientation = world;
      views[eye].pose.position = {position.x + side * right.x, position.y + side * right.y,
                                  position.z + side * right.z};
      views[eye].fov = {std::atan(g.shown_tangents[0]), std::atan(g.shown_tangents[1]),
                        std::atan(g.shown_tangents[2]), std::atan(g.shown_tangents[3])};
      views[eye].subImage.swapchain = g.swapchain;
      views[eye].subImage.imageRect = {{Stereo() ? eye * int32_t(eye_width) : 0, 0},
                                       {int32_t(eye_width), int32_t(g.height)}};
    }
    projection.space = g.local_space;
    projection.viewCount = 2;
    projection.views = views;
    layers[0] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&projection);
  }
  // The HUD: premultiplied over the scene, a 16:9 quad 1.4 m wide 1.1 m ahead of the seat (turned with it).
  XrCompositionLayerQuad hud{XR_TYPE_COMPOSITION_LAYER_QUAD};
  const XrCompositionLayerBaseHeader* all_layers[2] = {layers[0], reinterpret_cast<XrCompositionLayerBaseHeader*>(&hud)};
  const bool show_hud = g.shown_posed && g.shown_hud && g.hud_swapchain;
  if (show_hud) {
    const float yaw = g.reference_yaw, c = std::cos(yaw), s = std::sin(yaw);
    const float distance = 1.1f, width = 1.4f;
    hud.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    hud.space = g.local_space;
    hud.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    hud.subImage.swapchain = g.hud_swapchain;
    hud.subImage.imageRect = {{0, 0}, {int32_t(g.hud_width), int32_t(g.hud_height)}};
    hud.pose.orientation = {0.f, std::sin(yaw * 0.5f), 0.f, std::cos(yaw * 0.5f)};
    hud.pose.position = {g.reference_position.x - s * distance, g.reference_position.y,
                         g.reference_position.z - c * distance};
    hud.size = {width, width * g.hud_aspect.load(std::memory_order_relaxed)};
  }
  XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
  end.displayTime = g.frame_state.predictedDisplayTime;
  end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  end.layerCount = g.swapchain ? (show_hud ? 2 : 1) : 0;
  end.layers = all_layers;
  QueueLock lock;
  Check(g.xr.EndFrame(g.session, &end), "xrEndFrame");
}

bool Finished(const State::Published& p) {
  uint64_t value = 0;
  return g.vk.vkGetSemaphoreCounterValue(g.device, g.timeline, &value) != VK_SUCCESS || value >= p.tick;
}

// After xrWaitFrame: waits (up to 100 ms) for the game's next frame and its GPU work, so a slow game frame makes the
// headset frame late and the runtime sees it (Driveclub recomp c5c0dc6: the output rate is the game's).
// After xrWaitFrame: waits (up to 100 ms) until the GPU has finished a game frame newer than the one shown, so a slow
// game makes the headset frame late and the runtime sees it (Driveclub recomp c5c0dc6: the output rate is the
// game's). The ticks count the frames: the next one's tick is the shown one's plus one.
void WaitForGameFrame() {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  uint64_t next_tick = 0;
  {
    std::unique_lock lock{g.publish_mutex};
    if (!g.publish_cv.wait_until(lock, deadline, [] {
          return g.quit.load() || (g.recent[0].valid && g.recent[0].serial > g.shown_serial);
        })) {
      return;
    }
    // The oldest publication newer than the shown one.
    next_tick = g.recent[0].tick;
    for (const auto& p : g.recent) {
      if (p.valid && p.serial > g.shown_serial) {
        next_tick = std::min(next_tick, p.tick);
      }
    }
  }
  const auto left = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - std::chrono::steady_clock::now());
  if (left.count() <= 0) {
    return;
  }
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &g.timeline;
  info.pValues = &next_tick;
  g.vk.vkWaitSemaphores(g.device, &info, uint64_t(left.count()));
}

void FrameLoop() {
  while (!g.quit) {
    PollEvents();
    if (!g.running) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    static const bool trace = std::getenv("MCLA_VR_LOOP_TRACE") != nullptr;  // development: the loop's stages
    static uint32_t trace_n = 0;
    auto trace_t = std::chrono::steady_clock::now();
    XrTime trace_first_time = 0;
    double trace_ms[6] = {};
    auto stage = [&](int i) {
      const auto now = std::chrono::steady_clock::now();
      trace_ms[i] = std::chrono::duration<double, std::milli>(now - trace_t).count();
      trace_t = now;
    };
    // The frame cap (vr_frame_cap N): one headset frame per game frame, as RPCS3's frame thread does. The game is
    // paced to 1/N of the refresh (PaceSwap's timer), this loop waits for its next frame (up to 100 ms: loading,
    // pause) and only then takes a headset frame, so the runtime sees every other slot empty and fills it itself
    // (reprojection, or an external frame generator). Nothing is submitted for the slots between: begun-and-ended
    // repeats showed the last picture (Matt: no), layer-less ones black, a second xrWaitFrame blocked for good.
    if (std::clamp(REXCVAR_GET(vr_frame_cap), 1, 4) > 1) {
      std::unique_lock lock{g.publish_mutex};
      g.publish_cv.wait_for(lock, std::chrono::milliseconds(100), [] {
        return g.quit.load() || (g.recent[0].valid && g.recent[0].serial > g.shown_serial);
      });
    }
    g.frame_state = {XR_TYPE_FRAME_STATE};
    XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
    if (!Check(g.xr.WaitFrame(g.session, &wait, &g.frame_state), "xrWaitFrame")) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    trace_first_time = g.frame_state.predictedDisplayTime;
    stage(0);
    if (g.frame_state.predictedDisplayPeriod > 0) {
      const uint64_t period = uint64_t(g.frame_state.predictedDisplayPeriod);
      g.display_period_ns.store(period, std::memory_order_relaxed);
      if (!g.xr.GetDisplayRefreshRateFB) {
        const uint64_t shortest = g.refresh_period_ns.load(std::memory_order_relaxed);
        if (!shortest || period < shortest) {
          g.refresh_period_ns.store(period, std::memory_order_relaxed);
        }
      }
    }
    // With a frame cap the head pose and the token go out first, so the game draws its frame (for the display
    // time N periods on, the frame it will be shown in) while the N - 1 skipped frames are waited out; the wait for
    // its frame comes after. Token after the skip-wait (20:22, SteamVR, cap 2) left the game and this loop each
    // waiting on the other with 100 ms timeouts: 9.8 FPS, GPU idle, "Waiting..." in SteamVR.
    UpdateHead();
    {
      std::scoped_lock lock{g.token_mutex};
      g.flip_token.store(true, std::memory_order_relaxed);  // one game frame per headset frame (while capped)
    }
    g.token_cv.notify_all();
    stage(1);
    stage(2);
    WaitForGameFrame();
    stage(3);
    std::scoped_lock frame_lock{g.frame_mutex};
    {
      QueueLock lock;
      XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
      if (!Check(g.xr.BeginFrame(g.session, &begin), "xrBeginFrame")) {
        continue;
      }
    }
    decltype(g.recent) recent;
    {
      std::scoped_lock lock{g.publish_mutex};
      recent = g.recent;
    }
    bool shown = false;
    for (size_t i = 0; i < recent.size() && g.frame_state.shouldRender; ++i) {
      const State::Published& p = recent[i];
      if (!p.valid || p.serial <= g.shown_serial) {
        break;  // older ones were shown
      }
      if (i + 1 < recent.size() && Finished(p)) {  // the newest done (the oldest kept may be refilled: not it)
        g.shown_serial = p.serial;
        shown = Show(p);
        break;
      }
    }
    g.repeats += !shown;
    stage(4);
    EndFrame();
    stage(5);
    if (trace && ++trace_n % 90 == 0) {
      Log("loop trace: xrWaitFrame %.1f ms, head+token %.1f, skip-wait %.1f (display time %+.1f ms), game wait %.1f, "
          "show %.1f, xrEndFrame %.1f; shown %d, period %.2f ms",
          trace_ms[0], trace_ms[1], trace_ms[2],
          double(g.frame_state.predictedDisplayTime - trace_first_time) * 1e-6, trace_ms[3], trace_ms[4], trace_ms[5],
          int(shown), double(g.display_period_ns.load(std::memory_order_relaxed)) * 1e-6);
    }
    NoteFrameTime();
  }
}

// The game's frames, every 2 s: capped at one per headset frame once they hold 112% of the headset's rate, released
// below 95% (Driveclub's adaptive cap).
void NoteFlip() {
  static const bool enabled = !(std::getenv("MCLA_VR_CAP") && std::getenv("MCLA_VR_CAP")[0] == '0');
  const uint64_t refresh = g.refresh_period_ns.load(std::memory_order_relaxed);
  const uint64_t period = refresh ? refresh : g.display_period_ns.load(std::memory_order_relaxed);
  if (!enabled || !period) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (g.flips_since == std::chrono::steady_clock::time_point{}) {
    g.flips_since = now;
  }
  ++g.flips;
  const double seconds = std::chrono::duration<double>(now - g.flips_since).count();
  if (seconds < 2.0) {
    return;
  }
  const double rate = g.flips / seconds, hz = 1e9 / double(period);
  {  // slow frames: overscan (State::slow_frames)
    const bool slow = g.slow_frames.load(std::memory_order_relaxed);
    const double capped_hz = hz / std::clamp(REXCVAR_GET(vr_frame_cap), 1, 4);
    if (!slow && rate < 0.8 * capped_hz) {
      g.slow_frames = true;
      Log("slow frames (%.0f FPS at %.0f Hz)", rate, capped_hz);
    } else if (slow && rate > 0.95 * capped_hz) {
      g.slow_frames = false;
      Log("frames back at the headset's rate (%.0f FPS)", rate);
    }
  }
  // Capped always: a frame later than its slot just doesn't wait (PaceSwap), so a game slower than the headset loses
  // nothing, and a faster one makes one frame a headset frame.
  if (!g.capped.exchange(true, std::memory_order_relaxed)) {
    Log("the game's frames paced at the headset's %.0f Hz (it made %.0f FPS)", hz, rate);
  }
  g.flips = 0;
  g.flips_since = now;
}

}  // namespace

bool Active() { return g.instance != XR_NULL_HANDLE; }

const std::vector<const char*>& InstanceExtensions() {
  std::call_once(g.init_once, Init);
  return g.instance_extensions;
}

const std::vector<const char*>& DeviceExtensions(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                                 VkInstance instance, VkPhysicalDevice physical_device) {
  (void)get_instance_proc_addr;
  if (!Active() || g.device_extensions_queried) {
    return g.device_extensions;
  }
  g.device_extensions_queried = true;
  VkPhysicalDevice wanted{};
  if (Check(g.xr.GetVulkanGraphicsDeviceKHR(g.instance, g.system, instance, &wanted),
            "xrGetVulkanGraphicsDeviceKHR") &&
      wanted != physical_device) {
    Log("warning: the headset's GPU is not the one the renderer chose");
  }
  uint32_t size = 0;
  g.xr.GetVulkanDeviceExtensionsKHR(g.instance, g.system, 0, &size, nullptr);
  g.device_extension_text.resize(size);
  g.xr.GetVulkanDeviceExtensionsKHR(g.instance, g.system, size, &size, g.device_extension_text.data());
  g.device_extension_text.resize(size ? size - 1 : 0);
  g.device_extensions = Split(g.device_extension_text);
  dlss::AddDeviceExtensions(instance, physical_device, g.device_extensions);
  Log("Vulkan device extensions: %s", g.device_extension_text.c_str());
  if (Stereo()) {  // stereo: pixel shader textures pushed per draw, each eye its own (vr_twins PushTextures)
    auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        get_instance_proc_addr(instance, "vkEnumerateDeviceExtensionProperties"));
    uint32_t count = 0;
    enumerate(physical_device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    enumerate(physical_device, nullptr, &count, extensions.data());
    for (const VkExtensionProperties& extension : extensions) {
      if (!std::strcmp(extension.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        g.device_extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        auto properties_2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            get_instance_proc_addr(instance, "vkGetPhysicalDeviceProperties2"));
        VkPhysicalDevicePushDescriptorPropertiesKHR push{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &push;
        properties_2(physical_device, &properties);
        g.max_push_descriptors = push.maxPushDescriptors;
      }
    }
  }
  return g.device_extensions;
}

uint32_t QueueCount(PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkInstance instance,
                    VkPhysicalDevice physical_device, uint32_t queue_family, uint32_t used) {
  if (!Active() || (std::getenv("MCLA_VR_SHARED_QUEUE") && std::getenv("MCLA_VR_SHARED_QUEUE")[0] == '1')) {
    return used;
  }
  auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      get_instance_proc_addr(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  uint32_t count = 0;
  properties(physical_device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  properties(physical_device, &count, families.data());
  if (queue_family < count && families[queue_family].queueCount > used) {
    g.queue_index = used;
    return used + 1;
  }
  return used;
}

bool DesktopMirror() { return !Active() || REXCVAR_GET(vr_desktop_mirror); }

void OnDeviceCreated(const VulkanDevice* device, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                     VkInstance instance, VkPhysicalDevice physical_device, VkDevice vk_device,
                     uint32_t queue_family) {
  if (!Active()) {
    return;
  }
  // F8 (rebindable, bind_vr_recenter) recentres the view on the head; a SteamVR seated reset does too.
  rex::ui::RegisterBind("bind_vr_recenter", "F8", "Recentre the VR view", [] { Recenter(); });
  dlss::OnDeviceCreated(instance, physical_device, vk_device, get_instance_proc_addr);
  g.vulkan_device = device;
  g.device = vk_device;
  g.physical_device = physical_device;
  g.queue_family = queue_family;
  auto get_device_proc_addr =
      reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_instance_proc_addr(instance, "vkGetDeviceProcAddr"));
#define VK_FN(name) g.vk.name = reinterpret_cast<PFN_##name>(get_device_proc_addr(vk_device, #name));
  VK_FUNCTIONS
#undef VK_FN
  g.vk.vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
      get_instance_proc_addr(instance, "vkGetPhysicalDeviceMemoryProperties"));
  if (g.max_push_descriptors) {
    SetPushDescriptors(reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
                           get_device_proc_addr(vk_device, "vkCmdPushDescriptorSetKHR")),
                       g.max_push_descriptors);
  }
  if (!g.vk.vkGetSemaphoreCounterValue || !g.vk.vkWaitSemaphores) {
    Log("no timeline semaphores: VR off");
    g.xr.DestroyInstance(g.instance);
    g.instance = XR_NULL_HANDLE;
    return;
  }
  XrGraphicsRequirementsVulkanKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
  g.xr.GetVulkanGraphicsRequirementsKHR(g.instance, g.system, &requirements);
  XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
  binding.instance = instance;
  binding.physicalDevice = physical_device;
  binding.device = vk_device;
  binding.queueFamilyIndex = queue_family;
  g.own_queue = g.queue_index != 0;
  binding.queueIndex = g.queue_index;
  XrSessionCreateInfo info{XR_TYPE_SESSION_CREATE_INFO};
  info.next = &binding;
  info.systemId = g.system;
  {
    QueueLock lock;
    if (!Check(g.xr.CreateSession(g.instance, &info, &g.session), "xrCreateSession")) {
      g.session = XR_NULL_HANDLE;
      return;
    }
  }
  g.vk.vkGetDeviceQueue(vk_device, queue_family, g.queue_index, &g.queue);
  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = queue_family;
  g.vk.vkCreateCommandPool(vk_device, &pool_info, nullptr, &g.pool);
  VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = g.pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 3;
  g.vk.vkAllocateCommandBuffers(vk_device, &alloc, g.cmds);
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (VkFence& fence : g.fences) {
    g.vk.vkCreateFence(vk_device, &fence_info, nullptr, &fence);
  }
  VkSemaphoreTypeCreateInfo type_info{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphore_info.pNext = &type_info;
  g.vk.vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &g.timeline);
  XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
  space.poseInReferenceSpace.orientation.w = 1.f;
  Check(g.xr.CreateReferenceSpace(g.session, &space, &g.local_space), "xrCreateReferenceSpace");
  space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
  Check(g.xr.CreateReferenceSpace(g.session, &space, &g.view_space), "xrCreateReferenceSpace (view)");
  PickFormat();
  Log("session created (%s queue)", g.own_queue ? "its own" : "the renderer's");
  g.thread = std::thread{FrameLoop};
}

int SplitFrame(uint32_t width, uint32_t height, float aspect) {
  if (g.split_screen >= 0) {
    return g.split_screen;
  }
  g.hud_aspect.store(aspect, std::memory_order_relaxed);
  if (!g.session || g.format == VK_FORMAT_UNDEFINED || !width || !height) {
    return -1;
  }
  {
    // The frame being recorded has its pose at the front of the queue (VdSwap queued it, its swap packet takes it).
    // An empty queue: the GPU thread is ahead of the game's VdSwap (it runs the frame's draws as they are written), so
    // the pose is still to come, and the caller saw the frame's head-tracked scene. Refusing then left 1-5 frames in
    // 600 unsplit while driving, their HUD drawn into the head-tracked picture and missing from the HUD layer (Matt
    // 2026-10-26: "the HUD still flickers").
    std::scoped_lock lock{g.pose_mutex};
    if (!g.swap_poses.empty() && !g.swap_poses.front().posed) {
      // A frame the render thread draws again without a game update (MCLA with shadows off renders faster than it
      // updates: every other swap had no camera hook) is still drawn with the head-tracked camera: its scene started
      // with the camera's pose (SceneDrawStarted), so it splits as any other.
      std::scoped_lock camera_lock{g.camera_mutex};
      if (!g.drawn_posed) {
        return -1;
      }
    }
  }
  uint32_t out_width = width, out_height = height;
  dlss::OutputSize(width, height, out_width, out_height);
  if (!MakeScreens(out_width * (Stereo() ? 2 : 1), out_height, width, height)) {
    return -1;
  }
  g.split_screen = int(g.screen_next);
  g.screen_next = (g.screen_next + 1) % State::kScreens;
  return g.split_screen;
}

void HudCaptured() { g.split_hud_captured = true; }

void SetFrameHasScene(bool has_scene) { g.frame_has_scene = has_scene; }

int BeginFrame(uint32_t width, uint32_t height) {
  uint32_t out_width = width, out_height = height;
  dlss::OutputSize(width, height, out_width, out_height);
  if (!g.session || g.format == VK_FORMAT_UNDEFINED || !width || !height ||
      !MakeScreens(out_width * (Stereo() ? 2 : 1), out_height, width, height)) {
    g.split_screen = -1;
    return -1;
  }
  // A 2D frame right after head-tracked ones is left off the headset (it keeps the last posed frame): MCLA with
  // shadows off presents every other frame from a copy of the last scene without drawing one, and showing those on
  // the screen flickered between the world and the screen. A real 2D screen (a menu) shows from its third frame.
  {
    const bool posed = g.gpu_pose.posed && g.frame_has_scene;
    g.unposed_run = posed ? 0 : g.unposed_run + 1;
    if (!posed && g.last_shown_posed && g.unposed_run <= 2 && g.split_screen < 0) {
      return -1;
    }
    g.last_shown_posed = posed;
  }
  const bool split = g.split_screen >= 0;
  const uint32_t index = split ? uint32_t(g.split_screen) : g.screen_next;
  if (!split) {
    g.screen_next = (index + 1) % State::kScreens;
  }
  g.split_screen = -1;
  g.pending.valid = true;
  g.pending.hud = split && g.split_hud_captured;
  g.split_hud_captured = false;
  g.pending.posed = g.gpu_pose.posed && g.frame_has_scene;  // 2D only (a title screen): on the screen
  std::memcpy(g.pending.pose, g.gpu_pose.pose, sizeof(g.pending.pose));
  std::memcpy(g.pending.tangents, g.gpu_pose.tangents, sizeof(g.pending.tangents));
  g.pending.screen = index;
  g.pending.width = width;
  g.pending.height = height;
  g.pending.out_width = out_width;
  g.pending.out_height = out_height;
  return split ? -2 - int(index) : int(index);  // split (-2 - screen): the scene and the HUD are already copied
}

// A render target drawn at a resolution scale has its first row (column) unwritten, half a guest pixel's offset; the
// game's resolves fill it from the next one (resolve_resolution_scale_fill_half_pixel_offset), so do the copies.
void BlitEdges(VkCommandBuffer cmd, VkImage source, VkImage target, int32_t target_x, uint32_t width,
               uint32_t height, bool fill_column, bool fill_row) {
  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, EyeLayer(source), 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  if (fill_row && height > 1) {
    blit.srcOffsets[0] = {0, 1, 0};
    blit.srcOffsets[1] = {int32_t(width), 2, 1};
    blit.dstOffsets[0] = {target_x, 0, 0};
    blit.dstOffsets[1] = {target_x + int32_t(width), 1, 1};
    g.vk.vkCmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
  }
  if (fill_column && width > 1) {
    blit.srcOffsets[0] = {1, 0, 0};
    blit.srcOffsets[1] = {2, int32_t(height), 1};
    blit.dstOffsets[0] = {target_x, 0, 0};
    blit.dstOffsets[1] = {target_x + 1, int32_t(height), 1};
    g.vk.vkCmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
  }
}

void RecordFrameCopy(VkCommandBuffer cmd, VkImage image, VkImageLayout layout, uint32_t width, uint32_t height,
                     int screen, int kind_and_edges) {
  if (screen < 0 || screen >= int(State::kScreens) || !g.screen_images[screen]) {
    return;
  }
  const int kind = kind_and_edges & 0xFF;
  const bool fill_column = (kind_and_edges & kCopyFillColumn) != 0, fill_row = (kind_and_edges & kCopyFillRow) != 0;
  if (kind == kCopyToOutput) {  // the left eye's scene to the desktop (the guest output holds only the HUD)
    if (ReplayEye()) {
      return;
    }
    const VkImage scene = g.screen_images[screen];
    Barrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    // The middle band of the eye's view at the game's output aspect (the window shows the output at that aspect).
    float h = 1.f, v = 1.f;
    {
      std::scoped_lock lock{g.head_mutex};
      h = std::max(-g.head_tangents[0], g.head_tangents[1]);
      v = std::max(g.head_tangents[2], -g.head_tangents[3]);
    }
    const uint32_t screen_width = g.screen_width / (Stereo() ? 2 : 1), screen_height = g.screen_height;
    const uint32_t rows = std::min(screen_height, uint32_t(float(screen_height) * (h / std::max(v, 1e-3f)) *
                                                           g.hud_aspect.load(std::memory_order_relaxed)));
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0] = {0, int32_t((screen_height - rows) / 2), 0};
    blit.srcOffsets[1] = {int32_t(screen_width), int32_t((screen_height + rows) / 2), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {int32_t(width), int32_t(height), 1};
    g.vk.vkCmdBlitImage(cmd, scene, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit, VK_FILTER_LINEAR);
    Barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    return;
  }
  if (kind == kCopyHud) {  // the left eye's: both eyes draw the same HUD
    if (ReplayEye() || !g.hud_screen_images[screen]) {
      return;
    }
    const VkImage hud = g.hud_screen_images[screen];
    Barrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    Barrier(cmd, hud, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {int32_t(width), int32_t(height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {int32_t(width), int32_t(height), 1};
    g.vk.vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, hud, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                        &blit, VK_FILTER_NEAREST);
    BlitEdges(cmd, image, hud, 0, width, height, fill_column, fill_row);
    Barrier(cmd, hud, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    Barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
    return;
  }
  const VkImage target = g.screen_images[screen];
  // Stereo: each eye's replay copies its own picture (the right eye's is the guest output's twin) to its half; the
  // left eye's copy comes first in the same command buffer.
  const int eye = ReplayEye();
  if (eye) {
    image = TwinOf(image);
  }
  dlss::ApplyInPlace(cmd, image, layout, width, height, fill_column, fill_row);  // DLAA over the picture (vr_dlss)
  Barrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);
  Barrier(cmd, target, eye ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, eye ? VK_ACCESS_TRANSFER_READ_BIT : 0, VK_ACCESS_TRANSFER_WRITE_BIT);
  // The eye's half at the output size: DLSS's upscale into it, or (no DLSS this frame) the picture scaled up, its
  // unwritten first row / column left out.
  const uint32_t out_width = g.screen_width / (Stereo() ? 2 : 1), out_height = g.screen_height;
  const bool scaled = out_width != width || out_height != height;
  if (!scaled || !dlss::UpscaleInto(cmd, image, width, height, fill_column, fill_row, target,
                                    int32_t(eye * out_width), out_width, out_height)) {
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, EyeLayer(image), 1};
    blit.srcOffsets[0] = {scaled && fill_column ? 1 : 0, scaled && fill_row ? 1 : 0, 0};
    blit.srcOffsets[1] = {int32_t(width), int32_t(height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0] = {int32_t(eye * out_width), 0, 0};
    blit.dstOffsets[1] = {int32_t((eye + 1) * out_width), int32_t(out_height), 1};
    g.vk.vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, scaled ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    if (!scaled) {
      BlitEdges(cmd, image, target, int32_t(eye * width), width, height, fill_column, fill_row);
    }
  }
  Barrier(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  Barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_ACCESS_TRANSFER_READ_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
}

bool TakeFrameSignal(VkSemaphore& semaphore, uint64_t& value) {
  if (!g.pending.valid) {
    return false;
  }
  ++g.count_frames;
  NoteFlip();
  semaphore = g.timeline;
  value = ++g.timeline_value;
  {
    std::scoped_lock lock{g.publish_mutex};
    State::Published p = g.pending;
    p.tick = value;
    p.serial = ++g.published_serial;
    for (size_t i = g.recent.size() - 1; i > 0; --i) {
      g.recent[i] = g.recent[i - 1];
    }
    g.recent[0] = p;
  }
  g.pending = {};
  g.publish_cv.notify_all();
  return true;
}

// Capped: one game frame per headset refresh period, on the game thread's own schedule (a late frame starts it over).
// Waiting for the headset loop's turn instead put the game, GPU thread and GPU in series (71 FPS where 109 ran
// uncapped).
void PaceGameFrame() {}

// The game thread at VdSwap (OnGameSwap): the frame limiter's place, after the frame is handed to the GPU.
void PaceSwap() {
  static std::chrono::steady_clock::time_point next{};
  if (!Active() || !g.capped.load(std::memory_order_relaxed)) {
    next = {};
    return;
  }
  // The swap waits for the headset's next frame (its token; MCLA_VR_PACE_TOKEN=0: the timer below). The timer's
  // schedule of the refresh period drifted against the headset's frames: parked at 90 Hz, frames over a period
  // 2.4% -> 0.9% (2x4: 5.5% -> 0.6%), 1% low 79.5 -> 81.4 (2x4: 78 -> 83.6). Waiting here at the swap, unlike at the
  // frame timer before (0517679), does not put the game, GPU thread and GPU in series.
  static const bool token_pacing = [] {
    const char* v = std::getenv("MCLA_VR_PACE_TOKEN");
    return !(v && v[0] == '0');
  }();
  // At a frame cap (vr_frame_cap N) the game is paced by the timer below at N refresh periods, and the headset loop
  // follows the game (one headset frame per game frame): the token is the loop's clock only at cap 1.
  if (token_pacing && REXCVAR_GET(vr_frame_cap) <= 1) {
    std::unique_lock lock{g.token_mutex};
    g.token_cv.wait_for(lock, std::chrono::milliseconds(50), [] { return g.flip_token.load(std::memory_order_relaxed); });
    g.flip_token.store(false, std::memory_order_relaxed);
    return;
  }
  const uint64_t refresh = g.refresh_period_ns.load(std::memory_order_relaxed);
  const uint64_t period_ns = (refresh ? refresh : g.display_period_ns.load(std::memory_order_relaxed)) *
                             uint64_t(std::clamp(REXCVAR_GET(vr_frame_cap), 1, 4));  // the frame cap: 1/N
  if (!period_ns) {
    return;
  }
  const auto period = std::chrono::nanoseconds(period_ns);
  auto now = std::chrono::steady_clock::now();
  if (next == std::chrono::steady_clock::time_point{} || now > next + period) {
    next = now + period;
    return;
  }
  if (now < next) {
    // Wait most of the way, then spin. A plain sleep rounds up to 15.6 ms once Windows 11 ignores the process's
    // timer resolution (its window occluded, as behind the headset's): a high-resolution waitable timer does not.
    constexpr auto kSpin = std::chrono::microseconds(500);
    if (next - now > kSpin + std::chrono::microseconds(200)) {
#ifdef _WIN32
      thread_local HANDLE timer =
          CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
      if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(next - now - kSpin).count() / 100);
        if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) {
          WaitForSingleObject(timer, INFINITE);
        }
      } else {
        std::this_thread::sleep_for(next - now - kSpin);
      }
#else
      std::this_thread::sleep_for(next - now - kSpin);
#endif
    }
    while (std::chrono::steady_clock::now() < next) {
      std::this_thread::yield();
    }
  }
  next += period;
}

void Recenter() { g.recenter_requested = true; }

double RefreshHz() {
  const uint64_t refresh = g.refresh_period_ns.load(std::memory_order_relaxed);
  const uint64_t period = refresh ? refresh : g.display_period_ns.load(std::memory_order_relaxed);
  return period ? 1e9 / double(period) : 0.0;
}

bool ShowingGameplay() { return g.shown_posed && g.shown_hud; }  // the title is head-tracked too, without the HUD layer

void NoteCameraPose(const float (&pose)[16], const float (&tangents)[4]) {
  std::scoped_lock lock{g.camera_mutex};
  std::memcpy(g.camera_pose, pose, sizeof(g.camera_pose));
  std::memcpy(g.camera_tangents, tangents, sizeof(g.camera_tangents));
  g.camera_posed = true;
}

void SceneDrawStarted() {
  std::scoped_lock lock{g.camera_mutex};
  if (!g.camera_posed) {
    return;
  }
  std::memcpy(g.drawn_pose, g.camera_pose, sizeof(g.drawn_pose));
  std::memcpy(g.drawn_tangents, g.camera_tangents, sizeof(g.drawn_tangents));
  g.drawn_posed = true;
}

bool DrawnPose(float (&pose)[16]) {
  std::scoped_lock lock{g.camera_mutex};
  std::memcpy(pose, g.drawn_pose, sizeof(pose));
  return g.drawn_posed;
}

void TraceGpuCameraYaw(float yaw) {
  const float pose[16] = {0, 0, 0, 0, 0, 0, 0, 0, std::sin(yaw / 57.2958f), 0, std::cos(yaw / 57.2958f)};
  PoseTrace("3b GPU draws the scene", pose);
}

float EyeShift(int eye) {
  const float shift = g.eye_shift.load(std::memory_order_relaxed);
  return eye ? -shift : shift;
}

float StereoBaseline() {
  static const float world_scale = [] {
    const char* v = std::getenv("MCLA_VR_WORLD_SCALE");
    return v ? float(std::atof(v)) : 1.f;
  }();
  return g.ipd * world_scale;
}

void SetFramePose(const float (&pose)[16], const float (&tangents)[4]) {
  PoseTrace("2 render-thread SetFramePose", pose);
  std::scoped_lock lock{g.pose_mutex};
  std::memcpy(g.frame_pose, pose, sizeof(g.frame_pose));
  std::memcpy(g.frame_tangents, tangents, sizeof(g.frame_tangents));
  g.frame_posed = true;
}

// VR fork: the game's swaps so far (GameSwapCount).
std::atomic<uint64_t> g_game_swaps{0};
uint64_t GameSwapCount() { return g_game_swaps.load(std::memory_order_relaxed); }

namespace {
struct FrameInterpolationEntry {
  uint64_t frame = ~0ull;
  double state = 0.0;
  float t = 1.f;
};
std::mutex g_interpolation_mutex;
std::array<FrameInterpolationEntry, 64> g_interpolation;
}  // namespace

void SetFrameInterpolation(uint64_t frame, double state, float t) {
  std::scoped_lock lock{g_interpolation_mutex};
  g_interpolation[frame % g_interpolation.size()] = {frame, state, t};
}

bool FrameInterpolation(uint64_t frame, double& state, float& t) {
  std::scoped_lock lock{g_interpolation_mutex};
  const FrameInterpolationEntry& e = g_interpolation[frame % g_interpolation.size()];
  if (e.frame != frame) {
    return false;
  }
  state = e.state;
  t = e.t;
  return true;
}

void OnGameSwap(uint32_t frontbuffer) {
  g_game_swaps.fetch_add(1, std::memory_order_relaxed);
  if (!Active()) {
    return;
  }
  PaceSwap();
  std::scoped_lock lock{g.pose_mutex};
  State::SwapPose swap{g.frame_posed};
  swap.frontbuffer = frontbuffer & 0x1FFFFFFF;
  std::memcpy(swap.pose, g.frame_pose, sizeof(swap.pose));
  PoseTrace("3 VdSwap", swap.pose);
  std::memcpy(swap.tangents, g.frame_tangents, sizeof(swap.tangents));
  g.frame_posed = false;  // the next frame sets its own (a frame without the camera hook goes to the screen)
  g.swap_poses.push_back(swap);
  while (g.swap_poses.size() > 16) {
    g.swap_poses.pop_front();
  }
}

void OnGpuSwap(uint32_t frontbuffer) {
  if (!Active()) {
    return;
  }
  ++g_vr_gpu_swaps;
  frontbuffer &= 0x1FFFFFFF;
  std::scoped_lock lock{g.pose_mutex};
  // The pose queued for this packet's frontbuffer; entries before it are swaps that never reached the GPU (dropped).
  size_t match = g.swap_poses.size();
  for (size_t i = 0; i < g.swap_poses.size(); ++i) {
    if (g.swap_poses[i].frontbuffer == frontbuffer) {
      match = i;
      break;
    }
  }
  if (match == g.swap_poses.size()) {
    if (g.swap_poses.empty()) {
      g.gpu_pose.posed = false;
      return;
    }
    match = 0;  // no entry for this frontbuffer: in order (the old pairing)
  }
  if (match) {
    static uint32_t logged = 0;
    if (logged++ < 10) {
      Log("pose queue: %u swaps before this frontbuffer's dropped (they never reached the GPU)", uint32_t(match));
    }
  }
  g.gpu_pose = g.swap_poses[match];
  {  // the pose the frame's scene was actually drawn with (SceneDrawStarted)
    static const bool from_gpu = [] {
      const char* v = std::getenv("MCLA_VR_POSE_FROM_GPU");
      return !(v && v[0] == '0');
    }();
    std::scoped_lock camera_lock{g.camera_mutex};
    if (from_gpu && g.drawn_posed) {
      // Posed too when the swap had no camera hook but its scene was drawn head-tracked (a frame drawn again
      // without a game update, SplitFrame).
      g.gpu_pose.posed = true;
      std::memcpy(g.gpu_pose.pose, g.drawn_pose, sizeof(g.gpu_pose.pose));
      std::memcpy(g.gpu_pose.tangents, g.drawn_tangents, sizeof(g.gpu_pose.tangents));
    }
    g.drawn_posed = false;
  }
  PoseTrace("4 GPU swap", g.gpu_pose.pose);
  g.swap_poses.erase(g.swap_poses.begin(), g.swap_poses.begin() + ptrdiff_t(match) + 1);
}

bool HeadPose(float (&pose)[16], float (&tangents)[4]) {
  std::scoped_lock lock{g.head_mutex};
  if (!g.head_valid) {
    return false;
  }
  std::memcpy(pose, g.head_pose, sizeof(pose));
  PoseTrace("1 camera HeadPose", pose);
  std::memcpy(tangents, g.head_tangents, sizeof(tangents));
  return true;
}

}  // namespace rex::ui::vulkan::vr
