/**
 * @file        rex/ui/surface_ps5.h
 * @brief       Presentation surface for PlayStation 5 homebrew.
 *
 * The console has no window system. The Vulkan driver presents to the display
 * itself through VK_KHR_display, so the surface is just the display at a
 * chosen size; the presenter picks the display mode of that size when it
 * creates the Vulkan surface.
 */

#pragma once

#include <cstdint>

#include <rex/ui/surface.h>

namespace rex {
namespace ui {

class Ps5DisplaySurface final : public Surface {
 public:
  Ps5DisplaySurface(uint32_t width, uint32_t height) : width_(width), height_(height) {}

  TypeIndex GetType() const override { return kTypeIndex_Ps5Display; }

 protected:
  bool GetSizeImpl(uint32_t& width_out, uint32_t& height_out) const override {
    width_out = width_;
    height_out = height_;
    return true;
  }

 private:
  uint32_t width_;
  uint32_t height_;
};

}  // namespace ui
}  // namespace rex
