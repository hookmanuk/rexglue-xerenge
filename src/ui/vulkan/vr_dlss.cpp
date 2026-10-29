// VR fork: DLSS for the headset's eyes (vr_dlss.h).
#include <rex/ui/vulkan/vr_dlss.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/api.h>  // before the bridge header (its vulkan.h)
#include <rex/ui/vulkan/bbport_dlss_bridge.h>
#include <rex/ui/vulkan/vr_openxr.h>
#include <rex/ui/vulkan/vr_twins.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

REXCVAR_DEFINE_BOOL(vr_dlss, false, "VR", "DLSS in VR (needs bbport_dlss.dll and nvngx_dlss.dll beside the program)");
REXCVAR_DEFINE_BOOL(vr_dlss_bypass, false, "VR",
                    "Live: DLSS loaded but not used: no jitter, the render size to the headset (the VR settings' NO AA "
                    "and MSAA headset sizes, switchable to DLSS while playing)");
REXCVAR_DEFINE_DOUBLE(vr_dlss_output_scale, 1.0, "VR",
                      "DLSS's output to the headset at this scale of the render size (1: DLAA; 1.5: an upscale)");
REXCVAR_DEFINE_STRING(vr_dlss_model, "default", "VR",
                      "Live: DLSS's model (NGX render preset): default (NGX's choice per mode: K for DLAA to Balanced, "
                      "M for Performance), J or K (transformer), L or M (second transformer, lighter), E or F (the "
                      "older convolutional ones)");

namespace rex::ui::vulkan::vr::dlss {

namespace {

struct State {
  bool tried = false;
  const BbDlssApi* api = nullptr;
  bool ready = false;
} g;

void BridgeLog(int warning, const char* message) {
  if (warning) {
    REXLOG_WARN("VR DLSS: {}", message);
  } else {
    REXLOG_INFO("VR DLSS: {}", message);
  }
}

// The bridge, loaded once (VR with vr_dlss only).
const BbDlssApi* Bridge() {
  if (g.tried) {
    return g.api;
  }
  g.tried = true;
  if (!Active() || !REXCVAR_GET(vr_dlss)) {
    return nullptr;
  }
#ifdef _WIN32
  wchar_t exe[MAX_PATH]{};
  GetModuleFileNameW(nullptr, exe, MAX_PATH);
  std::wstring folder(exe);
  folder.resize(folder.find_last_of(L"\\/") + 1);
  const HMODULE module = LoadLibraryW((folder + L"bbport_dlss.dll").c_str());
  if (!module) {
    REXLOG_WARN("VR DLSS: bbport_dlss.dll not found beside the program: no DLSS");
    return nullptr;
  }
  const auto get_api = reinterpret_cast<BbDlssGetApiFn>(GetProcAddress(module, "BbDlssGetApi"));
  const BbDlssApi* api = get_api ? get_api() : nullptr;
  if (!api || api->abi < 3 || !api->SelectFeature) {
    REXLOG_WARN("VR DLSS: the bridge is too old (ABI {}, 3 needed: a feature per eye)", api ? api->abi : 0);
    return nullptr;
  }
  if (!api->Configure(folder.c_str(), folder.c_str(), BridgeLog)) {
    REXLOG_WARN("VR DLSS: the bridge could not be configured (nvngx_dlss.dll beside the program?)");
    return nullptr;
  }
  g.api = api;
  REXLOG_INFO("VR DLSS: bridge loaded (ABI {})", api->abi);
#endif
  return g.api;
}

void AddNames(uint32_t count, const VkExtensionProperties* extensions, std::vector<const char*>& names) {
  for (uint32_t i = 0; i < count; ++i) {
    const char* name = extensions[i].extensionName;
    if (std::none_of(names.begin(), names.end(), [name](const char* n) { return std::strcmp(n, name) == 0; })) {
      names.push_back(name);
    }
  }
}

namespace shaders {
const uint32_t kMotion[] = {
#include "shaders/vr_dlss_motion.inc"
};
}  // namespace shaders

#define VR_DLSS_FUNCTIONS             \
  VK_FN(vkCreateImage)                \
  VK_FN(vkGetImageMemoryRequirements) \
  VK_FN(vkAllocateMemory)             \
  VK_FN(vkBindImageMemory)            \
  VK_FN(vkCreateImageView)            \
  VK_FN(vkDestroyImageView)           \
  VK_FN(vkDestroyImage)               \
  VK_FN(vkFreeMemory)                 \
  VK_FN(vkCreateSampler)              \
  VK_FN(vkCreateShaderModule)         \
  VK_FN(vkDestroyShaderModule)        \
  VK_FN(vkCreateDescriptorSetLayout)  \
  VK_FN(vkCreatePipelineLayout)       \
  VK_FN(vkCreateComputePipelines)     \
  VK_FN(vkDeviceWaitIdle)             \
  VK_FN(vkCmdPipelineBarrier)         \
  VK_FN(vkCmdBlitImage)               \
  VK_FN(vkCmdBindPipeline)            \
  VK_FN(vkCmdPushConstants)           \
  VK_FN(vkCmdPushDescriptorSetKHR)    \
  VK_FN(vkCmdDispatch)
struct Device {
#define VK_FN(name) PFN_##name name = nullptr;
  VR_DLSS_FUNCTIONS
#undef VK_FN
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory{};
  VkSampler sampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout motion_set_layout = VK_NULL_HANDLE;
  VkPipelineLayout motion_layout = VK_NULL_HANDLE;
  VkPipeline motion_pipeline = VK_NULL_HANDLE;
} d;

bool LoadDevice(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                PFN_vkGetInstanceProcAddr get_instance_proc_addr, PFN_vkGetDeviceProcAddr get_device_proc_addr) {
  d.device = device;
  bool ok = true;
#define VK_FN(name) ok &= (d.name = reinterpret_cast<PFN_##name>(get_device_proc_addr(device, #name))) != nullptr;
  VR_DLSS_FUNCTIONS
#undef VK_FN
  auto memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
      get_instance_proc_addr(instance, "vkGetPhysicalDeviceMemoryProperties"));
  if (!ok || !memory_properties) {
    return false;
  }
  memory_properties(physical_device, &d.memory);
  VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
  sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkDescriptorSetLayoutBinding bindings[4] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // the object motion
  };
  VkDescriptorSetLayoutCreateInfo set_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_layout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  set_layout.bindingCount = 4;
  set_layout.pBindings = bindings;
  const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24 * sizeof(float)};
  VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout.setLayoutCount = 1;
  layout.pSetLayouts = &d.motion_set_layout;
  layout.pushConstantRangeCount = 1;
  layout.pPushConstantRanges = &range;
  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = sizeof(shaders::kMotion);
  module_info.pCode = shaders::kMotion;
  VkShaderModule module = VK_NULL_HANDLE;
  if (d.vkCreateSampler(device, &sampler, nullptr, &d.sampler) != VK_SUCCESS ||
      d.vkCreateDescriptorSetLayout(device, &set_layout, nullptr, &d.motion_set_layout) != VK_SUCCESS ||
      d.vkCreatePipelineLayout(device, &layout, nullptr, &d.motion_layout) != VK_SUCCESS ||
      d.vkCreateShaderModule(device, &module_info, nullptr, &module) != VK_SUCCESS) {
    return false;
  }
  VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline.stage.module = module;
  pipeline.stage.pName = "main";
  pipeline.layout = d.motion_layout;
  const bool made =
      d.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &d.motion_pipeline) == VK_SUCCESS;
  d.vkDestroyShaderModule(device, module, nullptr);
  return made;
}

// An image of our own (device local), with a colour view.
struct Image {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0, height = 0;
};

bool MakeImage(Image& out, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage) {
  if (out.image) {
    d.vkDestroyImageView(d.device, out.view, nullptr);
    d.vkDestroyImage(d.device, out.image, nullptr);
    d.vkFreeMemory(d.device, out.memory, nullptr);
    out = {};
  }
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.usage = usage;
  if (d.vkCreateImage(d.device, &info, nullptr, &out.image) != VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements requirements;
  d.vkGetImageMemoryRequirements(d.device, out.image, &requirements);
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = requirements.size;
  alloc.memoryTypeIndex = UINT32_MAX;
  for (uint32_t i = 0; i < d.memory.memoryTypeCount && alloc.memoryTypeIndex == UINT32_MAX; ++i) {
    if ((requirements.memoryTypeBits >> i & 1) &&
        (d.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      alloc.memoryTypeIndex = i;
    }
  }
  if (alloc.memoryTypeIndex == UINT32_MAX || d.vkAllocateMemory(d.device, &alloc, nullptr, &out.memory) != VK_SUCCESS ||
      d.vkBindImageMemory(d.device, out.image, out.memory, 0) != VK_SUCCESS) {
    return false;
  }
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = out.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = format;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (d.vkCreateImageView(d.device, &view, nullptr, &out.view) != VK_SUCCESS) {
    return false;
  }
  out.format = format;
  out.width = width;
  out.height = height;
  return true;
}

void Barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
             VkAccessFlags src, VkAccessFlags dst) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = src;
  b.dstAccessMask = dst;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {aspect, 0, 1, EyeLayer(image), 1};  // (multiview: the eye's layer)
  d.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
}

// Per eye: DLSS's inputs and output, the previous frame's camera, whether this frame's depth is in.
struct Eye {
  Image depth, motion, input, output;
  float previous_vp[16]{};
  bool have_previous = false;
  // Before this frame's first depth (a later scene resolve in the frame takes it again).
  float previous_vp_before[16]{};
  bool have_previous_before = false;
  uint32_t previous_tag = 0;  // the frame of the last depth and motion
  uint32_t previous_tag_before = 0;
  bool cut = false;
  bool reset = true;          // DLSS's history dropped (the first frame, after a gap or a camera cut)
  bool depth_ready = false;
  bool feature = false;
  // VR fork: the output size changed (the settings menu) with a feature made: the new one waits until the frames
  // using the old are done (CreateFeature releases the old at once), the plain copy meanwhile.
  bool feature_hold = false;
  uint32_t feature_hold_until = 0;
  uint32_t preset = 0;  // the feature's NGX render preset (0: NGX's default)
  float jitter_x = 0.f, jitter_y = 0.f;  // the frame's, render pixels (for DLSS)
};
std::array<Eye, 2> eyes;
uint32_t current_frame = 0;  // the frame EnsureEye last saw

}  // namespace

void OutputSize(uint32_t width, uint32_t height, uint32_t& output_width, uint32_t& output_height) {
  const double scale =
      g.ready && !REXCVAR_GET(vr_dlss_bypass) ? std::clamp(REXCVAR_GET(vr_dlss_output_scale), 1.0, 3.0) : 1.0;
  output_width = std::max(width, uint32_t(double(width) * scale + 0.5) & ~1u);
  output_height = std::max(height, uint32_t(double(height) * scale + 0.5) & ~1u);
}

namespace {

// Images replaced at a size change, destroyed 16 frames on (the submissions in flight are done with them by then; a
// device wait here, on the submission thread, would race the headset thread's queue).
struct Retired {
  Image image;
  uint32_t frame;
};
std::vector<Retired> retired;

void DestroyRetired(uint32_t frame) {
  for (size_t i = 0; i < retired.size();) {
    if ((frame - retired[i].frame) % 4096 >= 16) {
      Image& image = retired[i].image;
      d.vkDestroyImageView(d.device, image.view, nullptr);
      d.vkDestroyImage(d.device, image.image, nullptr);
      d.vkFreeMemory(d.device, image.memory, nullptr);
      retired[i] = retired.back();
      retired.pop_back();
    } else {
      ++i;
    }
  }
}

// vr_dlss_model as NGX's render preset: A is 1, B 2 and so on; 0 for NGX's default.
uint32_t Preset() {
  const std::string model = REXCVAR_GET(vr_dlss_model);
  return model.size() == 1 && std::toupper(model[0]) >= 'A' && std::toupper(model[0]) <= 'O'
             ? uint32_t(std::toupper(model[0]) - 'A' + 1)
             : 0;
}

bool EnsureEye(Eye& eye, uint32_t width, uint32_t height, uint32_t frame) {
  current_frame = frame;
  if (!retired.empty()) {
    DestroyRetired(frame);
  }
  if (eye.feature && eye.preset != Preset()) {  // the settings menu's model: made again, as for a new size
    eye.feature = false;
    eye.feature_hold = true;
    eye.feature_hold_until = frame + 16;
    eye.have_previous = false;
    REXLOG_INFO("VR DLSS: model {}: the new feature in 16 frames", REXCVAR_GET(vr_dlss_model));
  }
  uint32_t output_width, output_height;
  OutputSize(width, height, output_width, output_height);
  if (eye.depth.image && eye.depth.width == width && eye.depth.height == height && eye.output.width == output_width &&
      eye.output.height == output_height) {
    return true;
  }
  for (Image* image : {&eye.depth, &eye.motion, &eye.input, &eye.output}) {
    if (image->image) {
      retired.push_back({*image, frame});
      *image = {};
    }
  }
  if (eye.feature) {  // a new size while running: the old feature may be in frames still on the GPU
    eye.feature_hold = true;
    eye.feature_hold_until = frame + 16;
    REXLOG_INFO("VR DLSS: a new output size, {}x{}: the new feature in 16 frames", output_width, output_height);
  }
  eye.feature = false;
  eye.have_previous = false;
  const VkImageUsageFlags sampled_storage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  return MakeImage(eye.depth, VK_FORMAT_R32_SFLOAT, width, height, sampled_storage) &&
         MakeImage(eye.motion, VK_FORMAT_R16G16_SFLOAT, width, height, sampled_storage) &&
         MakeImage(eye.input, VK_FORMAT_R16G16B16A16_SFLOAT, width, height,
                   sampled_storage | VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
         MakeImage(eye.output, VK_FORMAT_R16G16B16A16_SFLOAT, output_width, output_height,  // cleared at a reset
                   sampled_storage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

// 4x4 row-major helpers (row vectors).
void Multiply(const float* a, const float* b, float* out) {
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      float sum = 0.f;
      for (int k = 0; k < 4; ++k) {
        sum += a[r * 4 + k] * b[k * 4 + c];
      }
      out[r * 4 + c] = sum;
    }
  }
}

bool Invert(const float* m, float* out) {
  double a[4][8];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      a[r][c] = m[r * 4 + c];
      a[r][c + 4] = r == c ? 1.0 : 0.0;
    }
  }
  for (int c = 0; c < 4; ++c) {
    int pivot = c;
    for (int r = c + 1; r < 4; ++r) {
      if (std::abs(a[r][c]) > std::abs(a[pivot][c])) {
        pivot = r;
      }
    }
    if (std::abs(a[pivot][c]) < 1e-12) {
      return false;
    }
    std::swap(a[c], a[pivot]);
    const double inv = 1.0 / a[c][c];
    for (double& v : a[c]) {
      v *= inv;
    }
    for (int r = 0; r < 4; ++r) {
      if (r != c) {
        const double f = a[r][c];
        for (int k = 0; k < 8; ++k) {
          a[r][k] -= f * a[c][k];
        }
      }
    }
  }
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      out[r * 4 + c] = float(a[r][c + 4]);
    }
  }
  return true;
}

// The GPU thread's: the latest scene draw's view-projection this frame.
struct SceneCamera {
  float vp[16]{};
  bool valid = false;
  uint32_t taken = 0;  // the frame its depth was taken
  // This frame's scene resolves with the first one's camera so far, its camera, and the last frame's count: the
  // depth is taken at that one (the frame's last) and any after it.
  uint32_t resolves = 0, last_frame_resolves = 0;
  float first_vp[16]{};
} scene;

// The object motion image the pixel shaders write (the right eye's: its twin), and each draw's world matrix by key
// this frame and the last (the GPU thread).
struct Objects {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  uint32_t width = 0, height = 0;
  struct Draw {
    float world[16];
  };
  std::unordered_map<uint64_t, std::vector<Draw>> current, previous;  // by key, in drawing order
  float frame_vp[16]{}, previous_vp[16]{};
  bool have_frame_vp = false, have_previous_vp = false;
  uint32_t tag = 1;
  uint32_t render_width = 0, render_height = 0;
} objects;

}  // namespace

void AddInstanceExtensions(std::vector<const char*>& names) {
  const BbDlssApi* api = Bridge();
  uint32_t count = 0;
  const VkExtensionProperties* extensions = nullptr;
  if (api && api->InstanceExtensions(&count, &extensions)) {
    AddNames(count, extensions, names);
  }
}

void AddDeviceExtensions(VkInstance instance, VkPhysicalDevice physical_device, std::vector<const char*>& names) {
  const BbDlssApi* api = Bridge();
  uint32_t count = 0;
  const VkExtensionProperties* extensions = nullptr;
  if (api && api->DeviceExtensions(instance, physical_device, &count, &extensions)) {
    AddNames(count, extensions, names);
  }
}

void OnDeviceCreated(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                     PFN_vkGetInstanceProcAddr get_instance_proc_addr) {
  const BbDlssApi* api = Bridge();
  if (!api) {
    return;
  }
  const auto get_device_proc_addr =
      reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_instance_proc_addr(instance, "vkGetDeviceProcAddr"));
  if (!api->Initialize(instance, physical_device, device, get_instance_proc_addr, get_device_proc_addr)) {
    REXLOG_WARN("VR DLSS: NGX could not start (an NVIDIA RTX GPU and a recent driver are needed): no DLSS");
    return;
  }
  if (!LoadDevice(instance, physical_device, device, get_instance_proc_addr, get_device_proc_addr)) {
    REXLOG_WARN("VR DLSS: Vulkan functions missing: no DLSS");
    return;
  }
  g.ready = true;
  REXLOG_INFO("VR DLSS: ready");
}

bool Ready() { return g.ready; }

namespace {
// Halton(2, 3): the sub-pixel offsets, -0.5..0.5 pixels.
float Halton(uint32_t index, uint32_t base) {
  float f = 1.f, r = 0.f;
  for (uint32_t i = index; i; i /= base) {
    f /= float(base);
    r += f * float(i % base);
  }
  return r;
}
struct Jitter {
  uint32_t index = 0;
  float pixels_x = 0.f, pixels_y = 0.f;  // render pixels
  float ndc_x = 0.f, ndc_y = 0.f;
} jitter;
}  // namespace

void NextFrame(uint32_t width, uint32_t height) {
  {  // VR fork, development: MCLA_VR_TEST_SCALE=frame:scale[,frame:scale...] sets the output scale at those frames
    // (a settings-menu change while running, without the menu)
    static uint32_t test_frame = 0;
    static const std::string test = std::getenv("MCLA_VR_TEST_SCALE") ? std::getenv("MCLA_VR_TEST_SCALE") : "";
    ++test_frame;
    size_t at = 0;
    while (!test.empty() && at < test.size()) {
      const size_t colon = test.find(':', at), comma = test.find(',', at);
      if (colon == std::string::npos) {
        break;
      }
      if (uint32_t(std::atoi(test.c_str() + at)) == test_frame) {
        const std::string scale = test.substr(colon + 1, comma == std::string::npos ? std::string::npos : comma - colon - 1);
        if (scale == "off" || scale == "on") {  // the bypass (the settings' NO AA / MSAA)
          rex::cvar::SetFlagByName("vr_dlss_bypass", scale == "off" ? "true" : "false");
        } else {
          rex::cvar::SetFlagByName("vr_dlss_output_scale", scale);
        }
        REXLOG_INFO("VR DLSS test: frame {}, {}", test_frame, scale);
      }
      at = comma == std::string::npos ? test.size() : comma + 1;
    }
  }
  if (!g.ready || !width || !height) {
    return;
  }
  if (scene.taken != objects.tag && scene.resolves) {  // the frame ended before the resolve its depth waited for
    static int count = 0;
    if (++count % 100 == 1) {
      REXLOG_INFO("VR DLSS: frame {} had {} scene resolves, the last frame {}: no depth ({} so far)", objects.tag,
                  scene.resolves, scene.last_frame_resolves, count);
    }
  }
  scene.last_frame_resolves = scene.resolves;
  scene.resolves = 0;
  // The draws' history: this frame's becomes the last.
  objects.previous.swap(objects.current);
  objects.current.clear();
  std::memcpy(objects.previous_vp, objects.frame_vp, sizeof(objects.frame_vp));
  objects.have_previous_vp = objects.have_frame_vp;
  objects.have_frame_vp = false;
  objects.tag = objects.tag % 4096 + 1;
  objects.render_width = width;
  objects.render_height = height;
  jitter.index = jitter.index % 8 + 1;  // 8 phases (DLSS's guidance for this scale)
  jitter.pixels_x = Halton(jitter.index, 2) - 0.5f;
  jitter.pixels_y = Halton(jitter.index, 3) - 0.5f;
  jitter.ndc_x = 2.f * jitter.pixels_x / float(width);
  jitter.ndc_y = 2.f * jitter.pixels_y / float(height);
  // VR fork: bypassed (the settings' NO AA / MSAA): no jitter; DLSS's history starts again when it is used again.
  static bool was_bypassed = false;
  const bool bypassed = REXCVAR_GET(vr_dlss_bypass);
  if (bypassed) {
    jitter.pixels_x = jitter.pixels_y = jitter.ndc_x = jitter.ndc_y = 0.f;
  }
  if (bypassed != was_bypassed) {
    for (Eye& eye : eyes) {
      eye.reset = true;
      eye.have_previous = false;
    }
    REXLOG_INFO("VR DLSS: {}", bypassed ? "bypassed (the render size, no DLSS)" : "in use again");
    was_bypassed = bypassed;
  }
}

void FrameJitter(float& x, float& y) {
  x = jitter.ndc_x;
  y = jitter.ndc_y;
}

void NoteSceneDraw(const float* world, const float* world_view_projection) {
  if (!g.ready) {
    return;
  }
  float world_inverse[16];
  if (Invert(world, world_inverse)) {
    Multiply(world_inverse, world_view_projection, scene.vp);
    scene.valid = true;
    std::memcpy(objects.frame_vp, scene.vp, sizeof(scene.vp));
    objects.have_frame_vp = true;
  }
}

void SetObjectMotionImage(VkImage image, VkImageView view, uint32_t width, uint32_t height) {
  objects.image = image;
  objects.view = view;
  objects.width = width;
  objects.height = height;
}

bool DrawMotion(uint64_t key, const float* world, const float* world_view_projection, EyePush& push) {
  static const bool off = std::getenv("MCLA_VR_OBJECT_MOTION") && std::getenv("MCLA_VR_OBJECT_MOTION")[0] == '0';
  if (!g.ready || off || !objects.image || objects.render_width > objects.width ||
      objects.render_height > objects.height) {
    return false;
  }
  std::vector<Objects::Draw>& draws = objects.current[key];
  const size_t index = draws.size();
  std::memcpy(draws.emplace_back().world, world, sizeof(Objects::Draw::world));
  const auto previous_draws = objects.previous.find(key);
  if (previous_draws == objects.previous.end() || !objects.have_previous_vp) {
    return false;
  }
  // Last frame's draw of this one: the same place among the key's draws when it has not moved (most of the city:
  // the scene's depth gives it the camera's motion), else the key's draw nearest where this one is, within a few
  // metres. The draws of one model (trees, lamps) share a key and come and go with culling while driving, so the
  // same place often holds another instance: its motion was the distance between the two (DLSS smeared trees).
  const std::vector<Objects::Draw>& candidates = previous_draws->second;
  const auto unmoved = [world](const Objects::Draw& d) {
    for (int i = 0; i < 16; ++i) {
      if (std::abs(d.world[i] - world[i]) > 1e-4f * (1.f + std::abs(world[i]))) {
        return false;
      }
    }
    return true;
  };
  if (index < candidates.size() && unmoved(candidates[index])) {
    return false;
  }
  const Objects::Draw* previous = nullptr;
  float nearest = 5.f * 5.f;  // metres, squared (a fast car moves ~1 m a frame)
  for (const Objects::Draw& d : candidates) {
    const float x = d.world[12] - world[12], y = d.world[13] - world[13], z = d.world[14] - world[14];
    if (const float distance = x * x + y * y + z * z; distance < nearest) {
      if (unmoved(d)) {
        return false;
      }
      nearest = distance;
      previous = &d;
    }
  }
  float inverse[16];
  if (!previous || !Invert(world_view_projection, inverse)) {
    return false;
  }
  // This frame's clip position to the previous frame's: back through this WVP, out through the last world and
  // view-projection.
  float previous_wvp[16];
  Multiply(previous->world, objects.previous_vp, previous_wvp);
  Multiply(inverse, previous_wvp, push.clip_to_prev);
  push.size_x = float(objects.render_width);
  push.size_y = float(objects.render_height);
  push.tag = float(objects.tag);
  return true;
}

bool TakeSceneCamera(float (&view_projection)[16], float& jitter_x, float& jitter_y, uint32_t& frame) {
  if (!g.ready || !scene.valid) {
    return false;
  }
  // A frame's scene resolves come after its depth prepass (~340 draws, no colour) and after the visible draws (~1200
  // more): the last one with the scene's camera is the one (the prepass's depth has no foliage the prepass leaves out,
  // and the visible draws' own motion is stored after it). A later pass with another camera is not the scene.
  // MCLA_VR_DEPTH_TAKE=first (development): the first only, as before.
  static const bool first_only = std::getenv("MCLA_VR_DEPTH_TAKE") && std::getenv("MCLA_VR_DEPTH_TAKE")[0] == 'f';
  if (scene.resolves) {
    bool same = !first_only;
    for (int i = 0; same && i < 16; ++i) {
      same = std::abs(scene.vp[i] - scene.first_vp[i]) <= 1e-5f * (1.f + std::abs(scene.first_vp[i]));
    }
    if (!same) {
      scene.valid = false;
      return false;
    }
  } else {
    std::memcpy(scene.first_vp, scene.vp, sizeof(scene.vp));
  }
  if (++scene.resolves < scene.last_frame_resolves) {  // a later one comes (as last frame)
    scene.valid = false;
    return false;
  }
  scene.taken = frame = objects.tag;
  std::memcpy(view_projection, scene.vp, sizeof(scene.vp));
  scene.valid = false;  // once: the next resolve needs scene draws of its own
  jitter_x = jitter.ndc_x;
  jitter_y = jitter.ndc_y;
  return true;
}

// A cut (the flyover's shots, a replay's cameras): the screen's centre, near and far, behind the last camera or
// more than a quarter of the screen from where it was.
bool CameraCut(const float* clip_to_prev) {
  for (const float z : {0.99f, 1.f}) {
    // Row vectors: (0, 0, z, 1) * M.
    const float w = z * clip_to_prev[11] + clip_to_prev[15];
    if (w <= 1e-6f) {
      return true;
    }
    const float x = (z * clip_to_prev[8] + clip_to_prev[12]) / w, y = (z * clip_to_prev[9] + clip_to_prev[13]) / w;
    if (x * x + y * y > 0.25f) {
      return true;
    }
  }
  return false;
}

void RecordDepthMotion(VkCommandBuffer cmd, VkImage depth_image, VkImageView depth_view, VkImageLayout layout,
                       uint32_t width, uint32_t height, const float* view_projection, float jitter_x,
                       float jitter_y, uint32_t frame) {
  const int index = ReplayEye();
  Eye& eye = eyes[index];
  if (!g.ready || !EnsureEye(eye, width, height, frame)) {
    return;
  }
  if (index) {  // the right eye's depth is the twin's
    depth_image = TwinOf(depth_image);
    depth_view = TwinViewOf(depth_view);
  }
  // This eye's view-projection: the centre's with the eye's clip shift (a constant on clip x: the translation row).
  float vp[16];
  std::memcpy(vp, view_projection, sizeof(vp));
  vp[12] += EyeShift(index);
  struct {
    float clip_to_prev[16];
    float size[2];
    float jitter[2];
    float tag;  // this frame's object motion entries
    float pad[3];
  } push{};
  push.tag = float(frame);
  float inverse[16];
  if (eye.have_previous && eye.previous_tag == frame) {  // a later depth this frame: from where the first started
    std::memcpy(eye.previous_vp, eye.previous_vp_before, sizeof(vp));
    eye.have_previous = eye.have_previous_before;
    eye.previous_tag = eye.previous_tag_before;
  }
  std::memcpy(eye.previous_vp_before, eye.previous_vp, sizeof(vp));
  eye.have_previous_before = eye.have_previous;
  const bool follows = eye.have_previous && eye.previous_tag % 4096 + 1 == frame;
  eye.previous_tag_before = eye.previous_tag;
  eye.cut = false;
  if (follows && Invert(vp, inverse)) {
    Multiply(inverse, eye.previous_vp, push.clip_to_prev);
    eye.reset = eye.cut = CameraCut(push.clip_to_prev);
  } else {
    for (int i = 0; i < 4; ++i) {
      push.clip_to_prev[i * 5] = 1.f;
    }
    eye.reset = true;
  }
  std::memcpy(eye.previous_vp, vp, sizeof(vp));
  eye.have_previous = true;
  eye.previous_tag = frame;
  push.size[0] = float(width);
  push.size[1] = float(height);
  push.jitter[0] = jitter_x;
  push.jitter[1] = jitter_y;
  // DLSS's jitter in render pixels (image y down: the guest's y up flips).
  static const float sign_x = std::getenv("MCLA_DLSS_JX") ? float(std::atof(std::getenv("MCLA_DLSS_JX"))) : 1.f;
  static const float sign_y = std::getenv("MCLA_DLSS_JY") ? float(std::atof(std::getenv("MCLA_DLSS_JY"))) : -1.f;
  eye.jitter_x = sign_x * jitter_x * 0.5f * float(width);
  eye.jitter_y = sign_y * jitter_y * 0.5f * float(height);

  const VkImageAspectFlags depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  Barrier(cmd, depth_image, depth_aspects, layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  Barrier(cmd, eye.depth.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
          VK_ACCESS_SHADER_WRITE_BIT);
  Barrier(cmd, eye.motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
          VK_ACCESS_SHADER_WRITE_BIT);
  d.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.motion_pipeline);
  const VkDescriptorImageInfo source{d.sampler, depth_view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
  const VkDescriptorImageInfo depth_out{VK_NULL_HANDLE, eye.depth.view, VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorImageInfo motion_out{VK_NULL_HANDLE, eye.motion.view, VK_IMAGE_LAYOUT_GENERAL};
  const VkDescriptorImageInfo object_in{VK_NULL_HANDLE, index ? TwinViewOf(objects.view) : objects.view,
                                        VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; ++i) {
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstBinding = uint32_t(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  }
  writes[0].pImageInfo = &source;
  writes[1].pImageInfo = &depth_out;
  writes[2].pImageInfo = &motion_out;
  writes[3].pImageInfo = &object_in;
  // The pixel shaders' object motion first (their stores before this pass's reads).
  {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    d.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b,
                           0, nullptr, 0, nullptr);
  }
  d.vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.motion_layout, 0, objects.view ? 4 : 3,
                              writes);
  d.vkCmdPushConstants(cmd, d.motion_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  d.vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
  Barrier(cmd, depth_image, depth_aspects, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, layout,
          VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  eye.depth_ready = true;
}

namespace {

// The eye's scene (`image`, in TRANSFER_SRC_OPTIMAL) through DLSS into the eye's output image (left in
// TRANSFER_SRC_OPTIMAL); false (nothing changed) without this frame's depth and motion.
bool EvaluateEye(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height, bool fill_column,
                 bool fill_row) {
  const int index = ReplayEye();
  Eye& eye = eyes[index];
  if (!g.ready || REXCVAR_GET(vr_dlss_bypass) || !eye.depth_ready || eye.depth.width != width ||
      eye.depth.height != height) {
    eye.depth_ready = false;
    return false;
  }
  eye.depth_ready = false;
  if (!eye.feature && eye.feature_hold) {
    if (int32_t(current_frame - eye.feature_hold_until) < 0) {
      return false;
    }
    eye.feature_hold = false;
  }
  if (!eye.feature) {
    g.api->SelectFeature(uint32_t(index));
    BbDlssFeature feature{};
    feature.input_width = width;
    feature.input_height = height;
    feature.output_width = eye.output.width;
    feature.output_height = eye.output.height;
    // DLAA at the render size; else the mode nearest the scale (Quality 1.5x, Balanced 1.72x, Performance 2x).
    const double scale = double(eye.output.width) / double(width);
    feature.quality = scale < 1.01 ? 0 : scale < 1.6 ? 1 : scale < 1.86 ? 2 : 3;
    feature.preset = eye.preset = Preset();
    if (!g.api->CreateFeature(cmd, &feature)) {
      REXLOG_WARN("VR DLSS: the eye's feature could not be made");
      return false;
    }
    eye.feature = true;
    REXLOG_INFO("VR DLSS: eye {} feature, {}x{} -> {}x{} (quality {}, model {})", index, width, height,
                eye.output.width, eye.output.height, feature.quality, REXCVAR_GET(vr_dlss_model));
  }
  // The scene into DLSS's input (its own format).
  VkImageBlit blit{};
  blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcSubresource.baseArrayLayer = EyeLayer(image);  // (multiview: the eye's layer)
  blit.srcOffsets[1] = blit.dstOffsets[1] = {int32_t(width), int32_t(height), 1};
  Barrier(cmd, eye.input.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
  // With the unwritten first row / column filled from the next (DLSS would blend them in).
  VkImageBlit blits[3] = {blit, blit, blit};
  uint32_t blit_count = 1;
  if (fill_row && height > 1) {
    blits[blit_count].srcOffsets[0] = {0, 1, 0};
    blits[blit_count].srcOffsets[1] = {int32_t(width), 2, 1};
    blits[blit_count].dstOffsets[0] = {0, 0, 0};
    blits[blit_count++].dstOffsets[1] = {int32_t(width), 1, 1};
  }
  if (fill_column && width > 1) {
    blits[blit_count].srcOffsets[0] = {1, 0, 0};
    blits[blit_count].srcOffsets[1] = {2, int32_t(height), 1};
    blits[blit_count].dstOffsets[0] = {0, 0, 0};
    blits[blit_count++].dstOffsets[1] = {1, int32_t(height), 1};
  }
  d.vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, eye.input.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit_count, blits, VK_FILTER_NEAREST);
  Barrier(cmd, eye.input.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  Barrier(cmd, eye.depth.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  Barrier(cmd, eye.motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  Barrier(cmd, eye.output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
          VK_ACCESS_SHADER_WRITE_BIT);
  const auto image_of = [](const Image& i) {
    BbDlssImage out{};
    out.image = i.image;
    out.view = i.view;
    out.range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    out.format = i.format;
    out.width = i.width;
    out.height = i.height;
    return out;
  };
  BbDlssEvaluate evaluate{};
  evaluate.color = image_of(eye.input);
  evaluate.depth = image_of(eye.depth);
  evaluate.motion = image_of(eye.motion);
  evaluate.output = image_of(eye.output);
  evaluate.jitter_x = eye.jitter_x;
  evaluate.jitter_y = eye.jitter_y;
  evaluate.reset = eye.reset ? 1 : 0;
  static int resets_logged = 0;
  if (eye.reset && !index && resets_logged < 40) {
    ++resets_logged;
    REXLOG_INFO("VR DLSS: history reset (frame {}, last depth at {}{})", eye.previous_tag, eye.previous_tag_before,
                eye.cut ? ", a cut" : "");
  }
  evaluate.frame_ms = 11.1f;
  g.api->SelectFeature(uint32_t(index));
  const bool ok = g.api->Evaluate(cmd, &evaluate) != 0;
  g.api->SelectFeature(0);
  static bool logged_first = false;
  if (!logged_first) {
    logged_first = true;
    REXLOG_INFO("VR DLSS: first evaluation {} (jitter {} {} px)", ok ? "done" : "failed", eye.jitter_x, eye.jitter_y);
  }
  if (!ok) {
    return false;
  }
  Barrier(cmd, eye.output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  return true;
}

}  // namespace

bool ApplyInPlace(VkCommandBuffer cmd, VkImage image, VkImageLayout layout, uint32_t width, uint32_t height,
                  bool fill_column, bool fill_row) {
  const Eye& eye = eyes[ReplayEye()];
  if (!g.ready || !eye.depth_ready || eye.output.width != width || eye.output.height != height) {
    return false;  // (an upscale goes through UpscaleInto)
  }
  Barrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  if (!EvaluateEye(cmd, image, width, height, fill_column, fill_row)) {
    Barrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    return false;
  }
  // The result back over the scene.
  VkImageBlit blit{};
  blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstSubresource.baseArrayLayer = EyeLayer(image);  // (multiview: the eye's layer)
  blit.srcOffsets[1] = blit.dstOffsets[1] = {int32_t(width), int32_t(height), 1};
  Barrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
  d.vkCmdBlitImage(cmd, eyes[ReplayEye()].output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
  Barrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout,
          VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
  return true;
}

bool UpscaleInto(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height, bool fill_column,
                 bool fill_row, VkImage target, int32_t target_x, uint32_t target_width, uint32_t target_height) {
  const Eye& eye = eyes[ReplayEye()];
  if (eye.output.width != target_width || eye.output.height != target_height ||
      !EvaluateEye(cmd, image, width, height, fill_column, fill_row)) {
    return false;
  }
  VkImageBlit blit{};
  blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {int32_t(target_width), int32_t(target_height), 1};
  blit.dstOffsets[0] = {target_x, 0, 0};
  blit.dstOffsets[1] = {target_x + int32_t(target_width), int32_t(target_height), 1};
  d.vkCmdBlitImage(cmd, eyes[ReplayEye()].output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
  return true;
}

}  // namespace rex::ui::vulkan::vr::dlss
