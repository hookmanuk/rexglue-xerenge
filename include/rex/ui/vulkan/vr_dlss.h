// VR fork: DLSS for the headset's eyes, through the MIT bridge bbport_dlss.dll (Driveclub's, ABI 3: a feature per
// eye). The program holds no NVIDIA code: the bridge (with nvngx_dlss.dll) is loaded from the program's folder when
// present and vr_dlss is on; without it nothing changes.
#pragma once

#include <vector>

#include <rex/ui/vulkan/api.h>
#include <rex/ui/vulkan/vr_twins.h>  // EyePush

namespace rex::ui::vulkan::vr::dlss {

/// The bridge's Vulkan instance extensions, added to `names` (before the instance is made). Loads the bridge.
void AddInstanceExtensions(std::vector<const char*>& names);
/// The bridge's device extensions for `physical_device`, added to `names`.
void AddDeviceExtensions(VkInstance instance, VkPhysicalDevice physical_device, std::vector<const char*>& names);
/// After the device is made: NGX's session.
void OnDeviceCreated(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                     PFN_vkGetInstanceProcAddr get_instance_proc_addr);
/// Whether DLSS can run (the bridge loaded, NGX initialised).
bool Ready();
/// The GPU thread at each swap: the next frame's jitter, for a picture of `width` x `height` (render pixels).
void NextFrame(uint32_t width, uint32_t height);
/// The frame's jitter in NDC (0 without DLSS), for the guest vertex shaders (the eye-shift push constant).
void FrameJitter(float& x, float& y);
/// The GPU thread, a perspective draw into the scene: RAGE's world (c0-c3) and world-view-projection (c8-c11), row
/// vectors. The frame's view-projection is world^-1 x WVP (the latest such draw's).
void NoteSceneDraw(const float* world, const float* world_view_projection);
/// The GPU thread at the scene's resolve: the frame's view-projection (row-major) and jitter (NDC), for the depth
/// pass; false when DLSS is off or no scene draw was seen.
bool TakeSceneCamera(float (&view_projection)[16], float& jitter_x, float& jitter_y, uint32_t& frame);
/// Replay (each eye): the scene's depth (a depth-aspect view of `depth_image` in `layout`) to the eye's DLSS depth,
/// with its camera motion (this frame's view-projection and jitter, the previous frame's kept per eye).
void RecordDepthMotion(VkCommandBuffer cmd, VkImage depth_image, VkImageView depth_view, VkImageLayout layout,
                       uint32_t width, uint32_t height, const float* view_projection, float jitter_x, float jitter_y,
                       uint32_t frame);
/// Setup: the object motion image the pixel shaders write (set 0, binding 2), `width` x `height` at least the render
/// size; its twin is the right eye's.
void SetObjectMotionImage(VkImage image, VkImageView view, uint32_t width, uint32_t height);
/// The GPU thread, a perspective scene draw (`key`: its shaders, vertices and indices): the same draw last frame (by
/// key and order) gives its own motion: push's clip_to_prev, size and tag filled; false when none.
bool DrawMotion(uint64_t key, const float* world, const float* world_view_projection, EyePush& push);
/// Replay (each eye), the finished scene `image` (in `layout`, `width` x `height`) about to go to the headset: DLSS
/// (DLAA) in place, when this eye's depth and motion are ready this frame. False when not applied.
/// fill_column / fill_row: the first column / row was left unwritten at a resolution scale (taken from the next).
bool ApplyInPlace(VkCommandBuffer cmd, VkImage image, VkImageLayout layout, uint32_t width, uint32_t height,
                  bool fill_column, bool fill_row);
/// The headset's eye size for a render size: vr_dlss_output_scale times it while DLSS runs, else the render size.
void OutputSize(uint32_t width, uint32_t height, uint32_t& output_width, uint32_t& output_height);
/// Replay (each eye), an upscale: the scene `image` (TRANSFER_SRC_OPTIMAL, `width` x `height`) through DLSS into
/// `target` (TRANSFER_DST_OPTIMAL) at x `target_x`, `target_width` x `target_height` (OutputSize). False when not
/// applied (nothing written).
bool UpscaleInto(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height, bool fill_column,
                 bool fill_row, VkImage target, int32_t target_x, uint32_t target_width, uint32_t target_height);

}  // namespace rex::ui::vulkan::vr::dlss
