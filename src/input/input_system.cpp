/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <mutex>
#include <algorithm>
#include <cmath>
#include <set>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include <rex/dbg.h>
#include <rex/input/device_assignment.h>
#include <rex/input/flags.h>
#include <rex/input/input_driver.h>
#include <rex/input/input_system.h>
#include <rex/input/mnk/mnk_input_driver.h>
#include <rex/input/nop/nop_input_driver.h>
#include <rex/input/sdl/sdl_input_driver.h>
#include <rex/input/state_merge.h>
#include <rex/input/xinput/xinput_input_driver.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(input_backend, "sdl", "Input", "Input backend: sdl, xinput")
    .allowed({"sdl", "xinput"});

REXCVAR_DEFINE_BOOL(guide_button, false, "Input", "Enable guide button pass-through");
namespace rex::input {

namespace {

// Synthetic devices are parked past every physical ordinal so they cannot push
// a real pad off guest user 0. SlotAssignment routes them by their synthetic
// flag and never reads this value.
constexpr uint32_t kSyntheticOrdinal = UINT32_MAX;

}  // namespace

InputSystem::InputSystem(rex::ui::Window* window) : window_(window) {}

InputSystem::~InputSystem() = default;

X_STATUS InputSystem::Setup() {
  return X_STATUS_SUCCESS;
}

void InputSystem::Shutdown() {
  // device_owners_ holds raw driver pointers.
  devices_.clear();
  device_owners_.clear();
  drivers_.clear();
}

void InputSystem::AddDriver(std::unique_ptr<InputDriver> driver) {
  drivers_.push_back(std::move(driver));
}

void InputSystem::AttachWindow(rex::ui::Window* window) {
  window_ = window;
  for (auto& driver : drivers_) {
    driver->OnWindowAvailable(window);
  }
}

void InputSystem::SetActiveCallback(std::function<bool()> callback) {
  for (auto& driver : drivers_) {
    driver->set_is_active_callback(callback);
  }
}

void InputSystem::SetDeviceAssignment(std::unique_ptr<DeviceAssignment> assignment) {
  assignment_ = std::move(assignment);
  if (assignment_) {
    assignment_->OnDevicesChanged(devices_);
  }
}

void InputSystem::RefreshDevices() {
  std::vector<DeviceInfo> seen;
  std::vector<InputDriver*> owners;
  std::vector<DeviceInfo> enumerated;
  for (auto& driver : drivers_) {
    enumerated.clear();
    driver->EnumerateDevices(enumerated);
    for (auto& info : enumerated) {
      seen.push_back(info);
      owners.push_back(driver.get());
    }
  }

  // Carry forward ordinals already handed out, so a device keeps its guest user
  // when another pad is unplugged.
  bool changed = seen.size() != devices_.size();
  std::vector<bool> fresh(seen.size(), false);
  for (size_t i = 0; i < seen.size(); i++) {
    auto existing = std::find_if(devices_.begin(), devices_.end(),
                                 [&](const DeviceInfo& d) { return d.id == seen[i].id; });
    if (existing != devices_.end()) {
      seen[i].ordinal = existing->ordinal;
      continue;
    }
    fresh[i] = true;
    changed = true;
  }

  // Runs after the carry-forward pass so a new device cannot take an ordinal a
  // live one is still holding. Lowest free rather than a growing counter,
  // because a reconnected pad arrives as a new device and would otherwise walk
  // off the end of the guest users.
  for (size_t i = 0; i < seen.size(); i++) {
    if (!fresh[i]) {
      continue;
    }
    // Only physical devices consume an ordinal, so the first pad to connect is
    // guest user 0 however many synthetic devices enumerated ahead of it.
    if (seen[i].synthetic) {
      seen[i].ordinal = kSyntheticOrdinal;
      fresh[i] = false;
      continue;
    }
    auto taken = [&](uint32_t candidate) {
      for (size_t j = 0; j < seen.size(); j++) {
        if (!fresh[j] && !seen[j].synthetic && seen[j].ordinal == candidate) {
          return true;
        }
      }
      return false;
    };
    uint32_t ordinal = 0;
    while (taken(ordinal)) {
      ordinal++;
    }
    seen[i].ordinal = ordinal;
    fresh[i] = false;
  }

  for (const auto& old : devices_) {
    if (std::none_of(seen.begin(), seen.end(),
                     [&](const DeviceInfo& d) { return d.id == old.id; })) {
      active_devices_.Forget(old.id);
      changed = true;
    }
  }

  std::vector<size_t> order(seen.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  // Stable: synthetic devices share one ordinal, and their relative order
  // decides which answers GetCapabilities when no pad is attached.
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return seen[a].ordinal < seen[b].ordinal; });

  devices_.clear();
  device_owners_.clear();
  for (size_t i : order) {
    devices_.push_back(seen[i]);
    device_owners_.push_back(owners[i]);
  }

  if (changed && assignment_) {
    assignment_->OnDevicesChanged(devices_);
  }
}

InputDriver* InputSystem::DriverForDevice(DeviceId id) {
  for (size_t i = 0; i < devices_.size(); i++) {
    if (devices_[i].id == id) {
      return device_owners_[i];
    }
  }
  return nullptr;
}

const DeviceInfo* InputSystem::DeviceInfoFor(DeviceId id) const {
  for (const auto& device : devices_) {
    if (device.id == id) {
      return &device;
    }
  }
  return nullptr;
}

X_RESULT InputSystem::GetCapabilities(uint32_t user_index, uint32_t flags,
                                      X_INPUT_CAPABILITIES* out_caps) {
  SCOPE_profile_cpu_f("hid");
  if (!out_caps || !assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);
  if (ids.empty()) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  // Prefer the pad in hand, so button glyphs follow it rather than whichever
  // device enumerated first.
  DeviceId chosen = active_devices_.Active(user_index);
  if (std::find(ids.begin(), ids.end(), chosen) == ids.end()) {
    chosen = ids.front();
  }

  auto* driver = DriverForDevice(chosen);
  if (!driver) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  return driver->GetDeviceCapabilities(chosen, flags, out_caps);
}

namespace {

// VR fork, development (MCLA_PAD_FILE=<file>): player one's pad from a text file, re-read every 50 ms, merged
// over the real pads: button names (a b x y start back lb rb ls rs up down left right), lt / rt (fully pressed),
// and lx= ly= rx= ry= (-1..1). Scripted runs drive the game without keys sent to the foreground window.
bool PadFileState(X_INPUT_GAMEPAD& pad) {
  static const char* path = std::getenv("MCLA_PAD_FILE");
  if (!path || !*path) {
    return false;
  }
  static std::mutex mutex;
  static uint64_t last_read = 0;
  static std::string text;
  std::scoped_lock lock{mutex};
  const uint64_t now = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  if (now - last_read >= 50) {
    last_read = now;
    text.clear();
    if (FILE* f = std::fopen(path, "rb")) {
      char buffer[512];
      const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, f);
      std::fclose(f);
      text.assign(buffer, n);
    }
  }
  static const struct {
    const char* name;
    uint16_t bit;
  } kButtons[] = {{"a", X_INPUT_GAMEPAD_A},           {"b", X_INPUT_GAMEPAD_B},
                  {"x", X_INPUT_GAMEPAD_X},           {"y", X_INPUT_GAMEPAD_Y},
                  {"start", X_INPUT_GAMEPAD_START},   {"back", X_INPUT_GAMEPAD_BACK},
                  {"lb", X_INPUT_GAMEPAD_LEFT_SHOULDER}, {"rb", X_INPUT_GAMEPAD_RIGHT_SHOULDER},
                  {"ls", X_INPUT_GAMEPAD_LEFT_THUMB}, {"rs", X_INPUT_GAMEPAD_RIGHT_THUMB},
                  {"up", X_INPUT_GAMEPAD_DPAD_UP},    {"down", X_INPUT_GAMEPAD_DPAD_DOWN},
                  {"left", X_INPUT_GAMEPAD_DPAD_LEFT}, {"right", X_INPUT_GAMEPAD_DPAD_RIGHT}};
  uint16_t buttons = 0;
  bool any = false;
  size_t at = 0;
  while (at < text.size()) {
    const size_t end = text.find_first_of(" \t\r\n,;", at);
    const std::string token = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
    at = end == std::string::npos ? text.size() : end + 1;
    if (token.empty()) {
      continue;
    }
    any = true;
    for (const auto& b : kButtons) {
      if (token == b.name) {
        buttons |= b.bit;
      }
    }
    if (token == "lt") {
      pad.left_trigger = 255;
    } else if (token == "rt") {
      pad.right_trigger = 255;
    } else if (token.size() > 3 && token[2] == '=') {
      const float v = std::clamp(float(std::atof(token.c_str() + 3)), -1.f, 1.f);
      const int16_t axis = int16_t(v * 32767.f);
      const std::string name = token.substr(0, 2);
      if (name == "lx") pad.thumb_lx = axis;
      if (name == "ly") pad.thumb_ly = axis;
      if (name == "rx") pad.thumb_rx = axis;
      if (name == "ry") pad.thumb_ry = axis;
    }
  }
  pad.buttons = uint16_t(uint16_t(pad.buttons) | buttons);
  return any || path;
}

}  // namespace

namespace {
void (*button_pressed_hook)(uint16_t button) = nullptr;
}  // namespace

void SetButtonPressedHook(void (*hook)(uint16_t button)) { button_pressed_hook = hook; }

X_RESULT InputSystem::GetState(uint32_t user_index, X_INPUT_STATE* out_state) {
  SCOPE_profile_cpu_f("hid");
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  X_INPUT_STATE merged = {};
  bool any = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_INPUT_STATE state = {};
    if (driver->GetDeviceState(id, &state) != X_ERROR_SUCCESS) {
      continue;
    }
    active_devices_.Observe(user_index, id, state.gamepad);
    if (state.gamepad.buttons != 0) {
      // Once per device: which one a press came from, when nobody expects one.
      static std::set<uint64_t> reported;
      if (reported.size() < 16 && reported.insert(uint64_t(id)).second) {
        const DeviceInfo* info = DeviceInfoFor(id);
        REXLOG_INFO("input: first press from device {} '{}'{}: buttons {:04X}", uint64_t(id),
                    info ? info->name : std::string("?"),
                    info && info->synthetic ? " (synthetic)" : "", uint32_t(state.gamepad.buttons));
      }
    }
    if (!any) {
      merged = state;
      any = true;
    } else {
      MergeInto(merged, state);
    }
  }

  if (user_index == 0 && PadFileState(merged.gamepad)) {  // VR fork: development
    static uint32_t packet = 0;
    merged.packet_number = ++packet;
    any = true;
  }
  if (user_index == 0) {  // VR fork: buttons going down (SetButtonPressedHook)
    static uint16_t buttons_were = 0;
    const uint16_t buttons = any ? uint16_t(merged.gamepad.buttons) : 0;
    const uint16_t pressed = buttons & ~buttons_were;
    buttons_were = buttons;
    for (uint16_t bit = 1; pressed && button_pressed_hook && bit; bit <<= 1) {
      if (pressed & bit) {
        button_pressed_hook(bit);
      }
    }
  }
  if (!any) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  if (out_state) {
    *out_state = merged;
  }
  return X_ERROR_SUCCESS;
}

namespace {
std::mutex g_vibration_observer_mutex;
VibrationObserver g_vibration_observer;
}  // namespace

void SetVibrationObserver(VibrationObserver observer) {
  std::lock_guard lock(g_vibration_observer_mutex);
  g_vibration_observer = std::move(observer);
}

X_RESULT InputSystem::SetState(uint32_t user_index, X_INPUT_VIBRATION* vibration) {
  SCOPE_profile_cpu_f("hid");
  // The host's say first (SetVibrationObserver): a vibration it takes reaches
  // the pads as a stop, and the title is told it worked - it takes a failure
  // to mean the motor kept its old value and would send it back for ever.
  bool taken = false;
  if (vibration) {
    std::lock_guard lock(g_vibration_observer_mutex);
    if (g_vibration_observer) {
      taken = g_vibration_observer(user_index, vibration->left_motor_speed,
                                   vibration->right_motor_speed);
    }
  }
  X_INPUT_VIBRATION stop{};
  if (taken) {
    vibration = &stop;
  }
  if (!assignment_) {
    return taken ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  // Every pad on this user belongs to the same player, so all of them buzz.
  // Only pads decide the result: synthetic devices accept any vibration and
  // would otherwise report success for a pad that never rumbled.
  bool any_pad = false;
  bool any_synthetic = false;
  bool pad_rumbled = false;
  X_RESULT pad_error = X_ERROR_DEVICE_NOT_CONNECTED;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    const DeviceInfo* info = DeviceInfoFor(id);
    if (!driver || !info) {
      continue;
    }
    X_RESULT result = driver->SetDeviceVibration(id, vibration);
    if (info->synthetic) {
      any_synthetic = true;
      continue;
    }
    any_pad = true;
    if (result == X_ERROR_SUCCESS) {
      pad_rumbled = true;
    } else {
      pad_error = result;
    }
  }
  if (taken) {
    return X_ERROR_SUCCESS;
  }
  if (any_pad) {
    return pad_rumbled ? X_ERROR_SUCCESS : pad_error;
  }
  return any_synthetic ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetKeystroke(uint32_t user_index, uint32_t flags,
                                   X_INPUT_KEYSTROKE* out_keystroke) {
  SCOPE_profile_cpu_f("hid");
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  bool any_connected = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_RESULT result = driver->GetDeviceKeystroke(id, flags, out_keystroke);
    if (result == X_ERROR_SUCCESS) {
      out_keystroke->user_index = static_cast<uint8_t>(user_index);
      return result;
    }
    if (result != X_ERROR_DEVICE_NOT_CONNECTED) {
      any_connected = true;
    }
  }
  return any_connected ? X_ERROR_EMPTY : X_ERROR_DEVICE_NOT_CONNECTED;
}

std::unique_ptr<InputSystem> CreateDefaultInputSystem(bool tool_mode) {
  auto input = std::make_unique<InputSystem>(nullptr);

  if (!tool_mode) {
#if REX_PLATFORM_WIN32
    if (REXCVAR_GET(input_backend) == "xinput") {
      auto xinput_driver = std::make_unique<xinput::XinputInputDriver>(nullptr, 0);
      if (xinput_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(xinput_driver));
      }
    }
#endif

    if (REXCVAR_GET(input_backend) == "sdl") {
      auto sdl_driver = std::make_unique<sdl::SDLInputDriver>(nullptr, 0);
      if (sdl_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(sdl_driver));
      }
    }

    // MnK driver (keyboard/mouse -> controller emulation)
    auto mnk_driver = std::make_unique<mnk::MnkInputDriver>(nullptr, 0);
    if (mnk_driver->Setup() == X_STATUS_SUCCESS) {
      input->AddDriver(std::move(mnk_driver));
    }
  }

  // NOP driver (primary in tool mode, fallback otherwise)
  uint8_t nop_index = tool_mode ? 0 : 1;
  input->AddDriver(std::make_unique<nop::NopInputDriver>(nullptr, nop_index));
  input->SetDeviceAssignment(std::make_unique<SlotAssignment>());
  return input;
}

}  // namespace rex::input
