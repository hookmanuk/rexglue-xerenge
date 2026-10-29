// VR fork: stereo by replaying each submission for the right eye.
//
// The renderer records every GPU command into its deferred command buffer; with stereo on, the submission replays it
// twice, the left eye on the renderer's own resources and the right eye on twins of them. Every image and buffer in
// device memory that the CPU cannot see (render targets, the EDRAM buffer, shared memory, textures, the guest output)
// gets a twin of the same size in a twin allocation, and so do the views, framebuffers, descriptor sets and query
// pools that use them. The right eye's replay goes through a function table whose command functions swap each handle
// for its twin, so resolves, render target transfers and post-processing all run per eye. Resources in host-visible
// memory (uploads, readbacks) stay shared: both eyes read the same uploads.

#pragma once

#include <vector>

#include <cstdint>

#include <rex/ui/vulkan/device.h>

namespace rex::ui::vulkan::vr {

/// Stereo: VR is on (Active) and MCLA_VR_STEREO is not 0.
bool Stereo();

/// Multiview stereo (Stereo, MCLA_VR_MULTIVIEW not 0): a render target has a layer per eye, and a pass into render
/// targets draws both eyes at once (view mask 0b11, the shaders taking the eye from ViewIndex). Everything outside
/// those passes (resolves, transfers to memory, texture loads, post-processing passes into other images) still runs
/// per eye, the right eye's on twins, a render target's twin being its second layer.
bool Multiview();
/// The render target cache, right before it makes a render target's image: that image gets a layer per eye.
void LayeredNextImage();
/// A view of a layered image (the left eye's layer) and the view of both its layers (2D array); null when not one.
VkImageView ArrayViewOf(VkImageView view);
/// The replay eye's layer of an image: the eye for a layered image, 0 otherwise.
uint32_t EyeLayer(VkImage image);
/// Multiview: a descriptor set bound in place of `set` inside passes that draw both eyes (its views of layered
/// images as views of both layers; the render target cache's transfer sources).
void SetMultiviewVariant(VkDescriptorSet set, VkDescriptorSet variant);
/// Multiview: the replay's function table for the left eye and for commands run once: a pass into layered targets
/// begins as multiview, and barriers on layered images cover the left eye's layer.
const VulkanDevice::Functions& LeftEyeFunctions();
/// Multiview, the replay thread: inside a pass drawing both eyes; whether the last pass begun did.
bool InMultiviewPass();
bool LastPassMultiview();
/// Multiview: a submission's replay begins and ends; the eye the following commands are replayed for.
void BeginReplay();
void EndReplay();
void SetReplayEye(int eye);

/// Device creation, its functions loaded: wraps the resource functions so device-local resources get twins.
void InstallTwins(VulkanDevice::Functions& functions, uint32_t host_visible_memory_types);
/// From the GPU thread, before it makes anything: only its resources get twins (the right eye replays its
/// submissions), and it reads the twins' maps without a lock (it is their only writer).
void SetGpuThread();

/// Stereo: guest pixel shader textures pushed per draw (VK_KHR_push_descriptor) rather than written to descriptor
/// sets, each eye's replay pushing its own views (no set twins, no second write on the GPU thread).
void SetPushDescriptors(PFN_vkCmdPushDescriptorSetKHR push_descriptor_set, uint32_t max_push_descriptors);
/// Whether a pixel shader's textures and samplers are pushed (multiview: each texture a pair, left and right).
bool PushTextures(size_t textures, size_t samplers);
/// The right eye's view of a texture: its twin when the picture differs per eye (GPU thread).
VkImageView RightEyeView(VkImageView view);
/// Replay: pushes `infos` (textures, samplers, then the right eye's textures) as set `set` of `layout`, the left or
/// right eye's textures by the replay's eye.
void PushTextureDescriptors(VkCommandBuffer command_buffer, VkPipelineLayout layout, uint32_t set,
                            uint32_t texture_count, uint32_t sampler_count, const VkDescriptorImageInfo* infos);

/// The right eye's function table: the device's, with the command functions using the twins.
const VulkanDevice::Functions& RightEyeFunctions();

/// The twin of an image (the image itself when it has none).
VkImage TwinOf(VkImage image);
/// The twin of a buffer (the buffer itself when it has none).
VkBuffer TwinBufferOf(VkBuffer buffer);

/// Stereo: lowers the draw resolution scale (keeping its shape where it can) when the estimate of what it needs (every
/// scaled resource twice) is over the device-local memory free now: past it the driver pages video memory to system
/// memory and the whole PC stalls. MCLA_VR_VRAM_GB_PER_AREA overrides the estimate's GB per unit of scale area.
void FitDrawResolutionScale(const VulkanDevice* device, uint32_t& scale_x, uint32_t& scale_y);
/// The GPU's device-local memory in GB (0 before the device starts), and the estimate of what stereo at a draw
/// resolution scale area needs (the guard's): for the settings menu.
double VideoMemoryTotalGB();
/// The device-local memory free now (budget less usage, VK_EXT_memory_budget; the total when it is missing).
double VideoMemoryFreeGB(const VulkanDevice* device);
double VideoMemoryNeededGB(uint32_t scale_area);

/// Development: device memory allocated for right-eye twins so far.
uint64_t TwinBytes();

/// The next sparse buffer this thread creates gets a sparse twin (else both eyes share a sparse buffer).
void SparseTwinNext();
/// Sparse binds: appends the twins' binds (each memory's twin) for buffers with a sparse twin. Every info's pBinds
/// holds an offset into `binds` (in and out); the caller turns them into pointers after the call.
void TwinSparseBinds(std::vector<VkSparseBufferMemoryBindInfo>& infos, std::vector<VkSparseMemoryBind>& binds);
/// The right eye's twin of an image view (the view itself when it has none).
VkImageView TwinViewOf(VkImageView view);

/// A texture: the same picture in both eyes (descriptors keep the left eye's view) until MarkEyeDependent, when it is
/// loaded from memory the GPU wrote (a resolve, different in each eye).
void MarkTexture(VkImage image);
void MarkEyeDependent(VkImage image);

/// Whether the view's image differs between the eyes (its right-eye descriptors use the twin).
bool IsViewEyeDependent(VkImageView view);
/// Whether any of the image infos' views is (one lock for all).
bool AnyViewEyeDependent(const VkDescriptorImageInfo* infos, size_t count);

/// The replay thread: before the right eye's replay (a barrier between the eyes; ReplayEye() is 1 until EndRightEye).
void BeginRightEye(VkCommandBuffer command_buffer);
void EndRightEye();

/// The eye this thread's replay records (0 left, 1 right).
int ReplayEye();

/// The push constant range of the guest draw pipeline layouts in stereo (the eye's clip-space shift, a float).
const VkPushConstantRange& EyeShiftRange();

/// A guest draw (deferred command kVrEyeShift): pushes the replay eye's shift for draws using `layout` (0 for a
/// draw that is not perspective: shadow maps, screen-space passes).
/// The guest shaders' push constant (vertex and fragment): the eye's shift, DLSS's jitter (NDC), and a draw's own
/// motion (motion 1: clip_to_prev, rows, maps its guest clip position to the previous frame's; the pixel shader stores
/// the motion, its depth and `tag` at its pixel of the object motion image, render pixels `size`).
struct EyePush {
  float shift, jitter_x, jitter_y, motion;
  float clip_to_prev[16];
  float size_x, size_y, tag;
  float shift_right;  // multiview: the right eye's shift (shift the left eye's)
};
void RecordEyeShift(VkCommandBuffer command_buffer, VkPipelineLayout layout, bool perspective,
                    const EyePush& push, const VulkanDevice::Functions& functions);

}  // namespace rex::ui::vulkan::vr
