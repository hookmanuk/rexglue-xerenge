// VR fork: OpenXR for the Vulkan renderer (ported from the Driveclub recomp's and the Uncharted port's VR layer).
//
// On with MCLA_VR=1 when an OpenXR runtime answers (XR_RUNTIME_JSON picks one, e.g. the OpenXR Simulator). The
// renderer calls in at a few points, marked "VR fork:": the instance's and the device's extensions, after the device
// is made, each game frame's finished picture (the guest output) and its submission.
//
// Stage 1: the game's picture on a screen floating in front of the player (a quad layer in the LOCAL space), paced
// as Driveclub's: the headset's loop waits for the game's next frame (so the runtime sees a slow game), and a game that
// holds more than the headset's rate is capped at one frame per headset frame (FlipAllowed).

#pragma once

#include <cstdint>
#include <vector>

#include <rex/ui/vulkan/api.h>

namespace rex::ui::vulkan {
class VulkanDevice;
}

namespace rex::ui::vulkan::vr {

/// MCLA_VR=1 and an OpenXR instance and headset exist (made by the first InstanceExtensions call).
bool Active();

/// The instance extensions the runtime needs (before vkCreateInstance). Makes the OpenXR instance on the first call.
const std::vector<const char*>& InstanceExtensions();

/// The device extensions the runtime needs for this physical device.
const std::vector<const char*>& DeviceExtensions(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                                 VkInstance instance, VkPhysicalDevice physical_device);

/// Device creation: queues to make in the graphics family (`used` of them the renderer's; one more for the headset's
/// own queue when the family has it).
/// Whether the desktop window is presented to: always outside VR; in VR only with vr_desktop_mirror. A windowed
/// present (the window behind the headset's) now and then stalled the GPU's graphics queue, and the game with it, for
/// good: no presents, no stalls (0 in 15 runs against ~1 in 6; a semaphore-free acquire or a queue of its own did not
/// help, the latter lost the device).
bool DesktopMirror();
uint32_t QueueCount(PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkInstance instance,
                    VkPhysicalDevice physical_device, uint32_t queue_family, uint32_t used);

/// After vkCreateDevice (with timeline semaphores enabled): the session and the headset's frame thread.
void OnDeviceCreated(const VulkanDevice* device, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                     VkInstance instance, VkPhysicalDevice physical_device, VkDevice vk_device,
                     uint32_t queue_family);

/// The GPU thread, recording the game's frame (its swap): the screen image its picture goes to (-1: none; -2 - screen:
/// the frame was split, its scene and HUD already copied to that screen). The submission that holds the copy (RecordFrameCopy) takes the
/// frame's signal (TakeFrameSignal).
int BeginFrame(uint32_t width, uint32_t height);
/// The GPU thread, before BeginFrame: whether the frame drew a 3D scene with the headset's field of view; one
/// without (2D only, or a camera of the game's own) is shown on the screen, not at its pose.
void SetFrameHasScene(bool has_scene);

/// The GPU thread, at the first HUD draw of a posed frame: the screen its scene goes to (each eye's render target
/// copied there, kCopyScene) before the target is cleared for the HUD, whose picture goes to the frame's HUD image
/// (kCopyHud, at the target's resolve). -1: no split (not posed, no headset).
int SplitFrame(uint32_t width, uint32_t height, float aspect);

/// The GPU thread: the split frame's HUD copy (kCopyHud) is recorded.
void HudCaptured();

// The scene's sharpening in the headset (0 off, to 1): plume's CAS per eye, as flat plume sharpens its frame (its own
// pass reads one layer and is skipped in stereo). Any thread; applied from the next frame shown.
void SetSceneSharpening(float strength);

/// RecordFrameCopy's kinds: the guest output; a render target holding the scene (each eye to its half); a render
/// target holding the HUD over transparent black (the left eye's, to the HUD image).
/// Or'ed in: the render target is drawn at a resolution scale in x (y): its first column (row) filled from the next.
/// kCopyToOutput: the other way, the left eye's scene into the guest output (a split frame's desktop picture).
enum CopyKind : int {
  kCopyFrame = 0,
  kCopyScene = 1,
  kCopyHud = 2,
  kCopyToOutput = 3,
  kCopyFillColumn = 0x100,
  kCopyFillRow = 0x200
};

/// The copy, at the submission's replay (its command buffer): `image` holds the finished picture (A2B10G10R10,
/// `layout`, `width` x `height`), copied into screen image `screen`, left in `layout`.
void RecordFrameCopy(VkCommandBuffer command_buffer, VkImage image, VkImageLayout layout, uint32_t width,
                     uint32_t height, int screen, int kind_and_edges = kCopyFrame);

/// The GPU thread, building the submission that holds RecordFrame's copy: the timeline semaphore and value it signals
/// when done. False: nothing to signal (no frame recorded since).
bool TakeFrameSignal(VkSemaphore& semaphore, uint64_t& value);

/// The game thread, before each frame: true when a frame may start now (uncapped, or the headset began a frame since
/// the last one). Waits for that while capped (at most 50 ms).
void PaceGameFrame();

/// The head's current position and yaw become the seat's origin (at the next frame).
void Recenter();
/// The headset's refresh rate in Hz (0 before the session has one), for the settings menu's frame cap labels.
double RefreshHz();

/// Whether the headset shows gameplay now: a head-tracked frame with the HUD on its own layer (not the title, menus
/// or loading).
bool ShowingGameplay();

/// The eye's (0 left, 1 right) shift of perspective geometry in clip space: x += P00 * (+-baseline / 2).
float EyeShift(int eye);

/// The distance between the eyes (the headset's IPD times MCLA_VR_WORLD_SCALE), in metres.
float StereoBaseline();

/// The game thread, its camera hook: the head pose (HeadPose's) and field of view the frame being built renders with.
void SetFramePose(const float (&pose)[16], const float (&tangents)[4]);

/// The game thread, VdSwap: the frame is complete; its pose goes with its swap packet, keyed by the frontbuffer (the
/// physical address the packet presents).
void OnGameSwap(uint32_t frontbuffer);
/// The game's swaps so far (OnGameSwap's calls): a game pacing in a function it also calls while loading (no swap)
/// paces only once per swap.
uint64_t GameSwapCount();

/// Drawing between logic steps (a game whose logic steps at a fixed rate under faster frames): the game, building
/// frame `frame` (its GameSwapCount()), says which logic state it draws (`state`, counting up a step at a time) and
/// where between the state before and it the frame's moment is (`t`, 0..1; 1: the state as it is). The GPU thread
/// reads it for the same frame (its swaps so far): FrameInterpolation, false when none was set.
void SetFrameInterpolation(uint64_t frame, double state, float t);
bool FrameInterpolation(uint64_t frame, double& state, float& t);

/// A renderer that draws the game's frames itself (not the xenos GPU thread: plume). At the frame's present, on the
/// thread that drew it: the pose its camera was read with (NoteCameraPose's since the last call; false: none, a 2D
/// frame).
bool TakeCameraPose(float (&pose)[16], float (&tangents)[4]);
/// Before BeginFrame for that frame: the pose it was drawn with (OnGpuSwap's part for xenos).
void SetPresentPose(bool posed, const float (&pose)[16], const float (&tangents)[4]);
/// The finished picture in `image` (`layers` layers: 2, an eye each, or 1 for both), each eye's to its half of screen
/// image `screen`; `image` left in `layout`.
void RecordLayeredFrameCopy(VkCommandBuffer command_buffer, VkImage image, VkImageLayout layout, uint32_t width,
                            uint32_t height, int screen, uint32_t layers);

/// The GPU thread, each swap packet (IssueSwap): the pose of the frame it presents, matched by its frontbuffer (by
/// order alone a swap queued at start without a packet left every later frame with the previous frame's pose: the
/// world moved when the head turned).
void OnGpuSwap(uint32_t frontbuffer);
/// Development (MCLA_VR_POSE_TRACE=1): the yaw the GPU draws a frame's scene with, logged when it changes.
void TraceGpuCameraYaw(float yaw);

/// The game's camera hook, when it turns the camera by a head pose (the update thread): the pose now in the camera.
void NoteCameraPose(const float (&pose)[16], const float (&tangents)[4]);
/// The GPU thread, a frame's first head-tracked scene draw: the camera's pose now is the one this frame is drawn with
/// (the GPU reads the camera the update thread wrote, a frame before the render thread's SetFramePose: labelling by
/// that showed every frame with the previous frame's pose, the world moving when the head turned). OnGpuSwap sends
/// the frame with it. MCLA_VR_POSE_FROM_GPU=0: the old labelling.
void SceneDrawStarted();
/// The GPU thread: the head pose SceneDrawStarted took for the frame being drawn (rows, as HeadPose). False before.
bool DrawnPose(float (&pose)[16]);

/// The head pose for the game's next frame: rows of the head's x, y, z axes and its position in the seat's space
/// (row vectors), and the eyes' field of view (tangents left, right, up, down). False before the first tracked frame.
bool HeadPose(float (&pose)[16], float (&tangents)[4]);
/// As HeadPose, carried on to now from the headset loop's last two poses: a game whose frames don't fall one to a
/// headset frame (60 Hz logic steps in some) gets a new pose for each.
bool HeadPoseNow(float (&pose)[16], float (&tangents)[4]);

}  // namespace rex::ui::vulkan::vr
