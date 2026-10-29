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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <rex/ui/vulkan/vr_openxr.h>
#include <rex/ui/vulkan/vr_dlss.h>
#include <rex/ui/vulkan/vr_twins.h>
#include <rex/assert.h>
#include <rex/dbg.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/math.h>

namespace rex::graphics::vulkan {

DeferredCommandBuffer::DeferredCommandBuffer(const VulkanCommandProcessor& command_processor,
                                             size_t initial_size)
    : command_processor_(command_processor) {
  command_stream_.reserve(initial_size / sizeof(uintmax_t));
}

void DeferredCommandBuffer::Reset() {
  command_stream_.clear();
  prologue_stream_.clear();
  prologue_ = false;
}

void DeferredCommandBuffer::BeginPrologue() {
  const bool first = prologue_stream_.empty();
  prologue_ = true;
  if (first) {  // after everything before the submission (its reads and writes), before the copies
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    CmdVkPipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
  }
}

void DeferredCommandBuffer::FinishPrologue() {
  prologue_ = false;
  if (prologue_stream_.empty()) {
    return;
  }
  {  // the copies before everything after them
    prologue_ = true;
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
    prologue_ = false;
  }
  prologue_stream_.insert(prologue_stream_.end(), command_stream_.begin(), command_stream_.end());
  command_stream_.swap(prologue_stream_);
  prologue_stream_.clear();
}

void DeferredCommandBuffer::Execute(VkCommandBuffer command_buffer,
                                    const std::vector<uintmax_t>& command_stream) const {
  if (ui::vulkan::vr::Multiview()) {
    ExecuteMultiview(command_buffer, command_stream);
    return;
  }
  ExecuteWith(command_buffer, command_stream, command_processor_.GetVulkanDevice()->functions());
  if (ui::vulkan::vr::Stereo()) {  // VR fork: the right eye replays the commands on its twins
    ui::vulkan::vr::BeginRightEye(command_buffer);
    ExecuteWith(command_buffer, command_stream, ui::vulkan::vr::RightEyeFunctions());
    ui::vulkan::vr::EndRightEye();
  }
}

// VR fork: multiview stereo. The stream in runs: the commands between passes, then a pass. A run between passes and
// a pass the left eye's table does not begin as multiview (not into render targets) are replayed for the left eye,
// then for the right on its twins; a pass into render targets draws both eyes at once. Compute state set inside such a
// pass (a dispatch's pipeline, sets and constants bound before the barriers that end the pass) is the left eye's
// there; the right eye's replay sets it again before its next run.
void DeferredCommandBuffer::ExecuteMultiview(VkCommandBuffer command_buffer,
                                             const std::vector<uintmax_t>& command_stream) const {
  const ui::vulkan::VulkanDevice::Functions& left = ui::vulkan::vr::LeftEyeFunctions();
  const ui::vulkan::VulkanDevice::Functions& right = ui::vulkan::vr::RightEyeFunctions();
  ui::vulkan::vr::BeginReplay();
  const uintmax_t* stream = command_stream.data();
  const uintmax_t* const end = stream + command_stream.size();
  const auto header_at = [](const uintmax_t* p) -> const CommandHeader& {
    return *reinterpret_cast<const CommandHeader*>(p);
  };
  const auto next = [&](const uintmax_t* p) {
    return p + kCommandHeaderSizeElements + header_at(p).arguments_size_elements;
  };
  const auto begins_pass = [&](const uintmax_t* p) {
    return header_at(p).command == Command::kVkBeginRendering ||
           header_at(p).command == Command::kVkBeginRenderPass;
  };
  const auto sets_compute_state = [&](const uintmax_t* p) {
    const void* args = p + kCommandHeaderSizeElements;
    switch (header_at(p).command) {
      case Command::kVkBindPipeline:
        return static_cast<const ArgsVkBindPipeline*>(args)->pipeline_bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
      case Command::kVkBindDescriptorSets:
        return static_cast<const ArgsVkBindDescriptorSets*>(args)->pipeline_bind_point ==
               VK_PIPELINE_BIND_POINT_COMPUTE;
      case Command::kVkPushConstants:
        return (static_cast<const ArgsVkPushConstants*>(args)->stage_flags & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
      default:
        return false;
    }
  };
  thread_local std::vector<const uintmax_t*> carried;  // compute state set in passes drawing both eyes
  carried.clear();
  while (stream < end) {
    const uintmax_t* run_end = stream;
    const bool pass = begins_pass(stream);
    if (pass) {
      const Command end_command = header_at(stream).command == Command::kVkBeginRendering
                                      ? Command::kVkEndRendering
                                      : Command::kVkEndRenderPass;
      while (run_end < end) {
        const bool ends = header_at(run_end).command == end_command;
        run_end = next(run_end);
        if (ends) {
          break;
        }
      }
    } else {
      while (run_end < end && !begins_pass(run_end)) {
        run_end = next(run_end);
      }
    }
    const size_t size = size_t(run_end - stream);
    ExecuteRange(command_buffer, stream, size, left);
    if (!pass || !ui::vulkan::vr::LastPassMultiview()) {
      ui::vulkan::vr::SetReplayEye(1);
      for (const uintmax_t* command : carried) {
        ExecuteRange(command_buffer, command, size_t(next(command) - command), right);
      }
      carried.clear();
      ExecuteRange(command_buffer, stream, size, right);
      ui::vulkan::vr::SetReplayEye(0);
    } else {
      for (const uintmax_t* p = stream; p < run_end; p = next(p)) {
        if (sets_compute_state(p)) {
          carried.push_back(p);
        }
      }
    }
    stream = run_end;
  }
  ui::vulkan::vr::EndReplay();
}

void DeferredCommandBuffer::ExecuteRange(VkCommandBuffer command_buffer, const uintmax_t* stream,
                                         size_t stream_remaining,
                                         const ui::vulkan::VulkanDevice::Functions& dfn) const {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  while (stream_remaining) {
    const CommandHeader& header = *reinterpret_cast<const CommandHeader*>(stream);
    stream += kCommandHeaderSizeElements;
    stream_remaining -= kCommandHeaderSizeElements;

    switch (header.command) {
      case Command::kVkBeginRenderPass: {
        auto& args = *reinterpret_cast<const ArgsVkBeginRenderPass*>(stream);
        size_t offset_bytes = sizeof(ArgsVkBeginRenderPass);
        VkRenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.pNext = nullptr;
        render_pass_begin_info.renderPass = args.render_pass;
        render_pass_begin_info.framebuffer = args.framebuffer;
        render_pass_begin_info.renderArea = args.render_area;
        render_pass_begin_info.clearValueCount = args.clear_value_count;
        if (render_pass_begin_info.clearValueCount) {
          offset_bytes = rex::align(offset_bytes, alignof(VkClearValue));
          render_pass_begin_info.pClearValues = reinterpret_cast<const VkClearValue*>(
              reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          offset_bytes += sizeof(VkClearValue) * render_pass_begin_info.clearValueCount;
        } else {
          render_pass_begin_info.pClearValues = nullptr;
        }
        dfn.vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, args.contents);
      } break;

      case Command::kVkBeginQuery: {
        auto& args = *reinterpret_cast<const ArgsVkBeginQuery*>(stream);
        dfn.vkCmdBeginQuery(command_buffer, args.query_pool, args.query, args.flags);
      } break;

      case Command::kVkBindDescriptorSets: {
        auto& args = *reinterpret_cast<const ArgsVkBindDescriptorSets*>(stream);
        size_t offset_bytes =
            rex::align(sizeof(ArgsVkBindDescriptorSets), alignof(VkDescriptorSet));
        const VkDescriptorSet* descriptor_sets = reinterpret_cast<const VkDescriptorSet*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes += sizeof(VkDescriptorSet) * args.descriptor_set_count;
        const uint32_t* dynamic_offsets = nullptr;
        if (args.dynamic_offset_count) {
          offset_bytes = rex::align(offset_bytes, alignof(uint32_t));
          dynamic_offsets = reinterpret_cast<const uint32_t*>(
              reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          offset_bytes += sizeof(uint32_t) * args.dynamic_offset_count;
        }
        dfn.vkCmdBindDescriptorSets(command_buffer, args.pipeline_bind_point, args.layout,
                                    args.first_set, args.descriptor_set_count, descriptor_sets,
                                    args.dynamic_offset_count, dynamic_offsets);
      } break;

      case Command::kVkBindIndexBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkBindIndexBuffer*>(stream);
        dfn.vkCmdBindIndexBuffer(command_buffer, args.buffer, args.offset, args.index_type);
      } break;

      case Command::kVkBindPipeline: {
        auto& args = *reinterpret_cast<const ArgsVkBindPipeline*>(stream);
        dfn.vkCmdBindPipeline(command_buffer, args.pipeline_bind_point, args.pipeline);
      } break;

      case Command::kVkBindVertexBuffers: {
        auto& args = *reinterpret_cast<const ArgsVkBindVertexBuffers*>(stream);
        size_t offset_bytes = rex::align(sizeof(ArgsVkBindVertexBuffers), alignof(VkBuffer));
        const VkBuffer* buffers = reinterpret_cast<const VkBuffer*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes =
            rex::align(offset_bytes + sizeof(VkBuffer) * args.binding_count, alignof(VkDeviceSize));
        const VkDeviceSize* offsets = reinterpret_cast<const VkDeviceSize*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        dfn.vkCmdBindVertexBuffers(command_buffer, args.first_binding, args.binding_count, buffers,
                                   offsets);
      } break;

      case Command::kVkClearAttachments: {
        auto& args = *reinterpret_cast<const ArgsVkClearAttachments*>(stream);
        size_t offset_bytes =
            rex::align(sizeof(ArgsVkClearAttachments), alignof(VkClearAttachment));
        const VkClearAttachment* attachments = reinterpret_cast<const VkClearAttachment*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes = rex::align(offset_bytes + sizeof(VkClearAttachment) * args.attachment_count,
                                  alignof(VkClearRect));
        const VkClearRect* rects = reinterpret_cast<const VkClearRect*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        dfn.vkCmdClearAttachments(command_buffer, args.attachment_count, attachments,
                                  args.rect_count, rects);
      } break;

      case Command::kVkClearColorImage: {
        auto& args = *reinterpret_cast<const ArgsVkClearColorImage*>(stream);
        dfn.vkCmdClearColorImage(
            command_buffer, args.image, args.image_layout, &args.color, args.range_count,
            reinterpret_cast<const VkImageSubresourceRange*>(
                reinterpret_cast<const uint8_t*>(stream) +
                rex::align(sizeof(ArgsVkClearColorImage), alignof(VkImageSubresourceRange))));
      } break;

      case Command::kVkCopyBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkCopyBuffer*>(stream);
        dfn.vkCmdCopyBuffer(command_buffer, args.src_buffer, args.dst_buffer, args.region_count,
                            reinterpret_cast<const VkBufferCopy*>(
                                reinterpret_cast<const uint8_t*>(stream) +
                                rex::align(sizeof(ArgsVkCopyBuffer), alignof(VkBufferCopy))));
      } break;

      case Command::kVkCopyBufferToImage: {
        auto& args = *reinterpret_cast<const ArgsVkCopyBufferToImage*>(stream);
        dfn.vkCmdCopyBufferToImage(
            command_buffer, args.src_buffer, args.dst_image, args.dst_image_layout,
            args.region_count,
            reinterpret_cast<const VkBufferImageCopy*>(
                reinterpret_cast<const uint8_t*>(stream) +
                rex::align(sizeof(ArgsVkCopyBufferToImage), alignof(VkBufferImageCopy))));
      } break;

      case Command::kVkCopyQueryPoolResults: {
        auto& args = *reinterpret_cast<const ArgsVkCopyQueryPoolResults*>(stream);
        dfn.vkCmdCopyQueryPoolResults(command_buffer, args.query_pool, args.first_query,
                                      args.query_count, args.dst_buffer, args.dst_offset,
                                      args.stride, args.flags);
      } break;

      case Command::kVkDispatch: {
        auto& args = *reinterpret_cast<const ArgsVkDispatch*>(stream);
        dfn.vkCmdDispatch(command_buffer, args.group_count_x, args.group_count_y,
                          args.group_count_z);
      } break;

      case Command::kVkDraw: {
        auto& args = *reinterpret_cast<const ArgsVkDraw*>(stream);
        dfn.vkCmdDraw(command_buffer, args.vertex_count, args.instance_count, args.first_vertex,
                      args.first_instance);
      } break;

      case Command::kVkDrawIndexed: {
        auto& args = *reinterpret_cast<const ArgsVkDrawIndexed*>(stream);
        dfn.vkCmdDrawIndexed(command_buffer, args.index_count, args.instance_count,
                             args.first_index, args.vertex_offset, args.first_instance);
      } break;

      case Command::kVkEndQuery: {
        auto& args = *reinterpret_cast<const ArgsVkEndQuery*>(stream);
        dfn.vkCmdEndQuery(command_buffer, args.query_pool, args.query);
      } break;

      case Command::kVkEndRenderPass:
        dfn.vkCmdEndRenderPass(command_buffer);
        break;

      case Command::kVkBeginRendering: {
        auto& args = *reinterpret_cast<const ArgsVkBeginRendering*>(stream);
        size_t offset_bytes =
            rex::align(sizeof(ArgsVkBeginRendering), alignof(VkRenderingAttachmentInfo));
        const VkRenderingAttachmentInfo* color_attachments =
            args.color_attachment_count
                ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                      reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                : nullptr;
        offset_bytes += sizeof(VkRenderingAttachmentInfo) * args.color_attachment_count;
        const VkRenderingAttachmentInfo* depth_attachment =
            args.has_depth_attachment ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                                            reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                                      : nullptr;
        if (args.has_depth_attachment) {
          offset_bytes += sizeof(VkRenderingAttachmentInfo);
        }
        const VkRenderingAttachmentInfo* stencil_attachment =
            args.has_stencil_attachment
                ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                      reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                : nullptr;
        VkRenderingInfo rendering_info = {};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.flags = args.flags;
        rendering_info.renderArea = args.render_area;
        rendering_info.layerCount = args.layer_count;
        rendering_info.viewMask = args.view_mask;
        rendering_info.colorAttachmentCount = args.color_attachment_count;
        rendering_info.pColorAttachments = color_attachments;
        rendering_info.pDepthAttachment = depth_attachment;
        rendering_info.pStencilAttachment = stencil_attachment;
        dfn.vkCmdBeginRendering(command_buffer, &rendering_info);
      } break;

      case Command::kVkEndRendering:
        dfn.vkCmdEndRendering(command_buffer);
        break;

      case Command::kVkPipelineBarrier: {
        auto& args = *reinterpret_cast<const ArgsVkPipelineBarrier*>(stream);
        size_t barrier_offset_bytes = sizeof(ArgsVkPipelineBarrier);
        const VkMemoryBarrier* memory_barriers = nullptr;
        if (args.memory_barrier_count) {
          barrier_offset_bytes = rex::align(barrier_offset_bytes, alignof(VkMemoryBarrier));
          memory_barriers = reinterpret_cast<const VkMemoryBarrier*>(
              reinterpret_cast<const uint8_t*>(stream) + barrier_offset_bytes);
          barrier_offset_bytes += sizeof(VkMemoryBarrier) * args.memory_barrier_count;
        }
        const VkBufferMemoryBarrier* buffer_memory_barriers = nullptr;
        if (args.buffer_memory_barrier_count) {
          barrier_offset_bytes = rex::align(barrier_offset_bytes, alignof(VkBufferMemoryBarrier));
          buffer_memory_barriers = reinterpret_cast<const VkBufferMemoryBarrier*>(
              reinterpret_cast<const uint8_t*>(stream) + barrier_offset_bytes);
          barrier_offset_bytes += sizeof(VkBufferMemoryBarrier) * args.buffer_memory_barrier_count;
        }
        const VkImageMemoryBarrier* image_memory_barriers = nullptr;
        if (args.image_memory_barrier_count) {
          barrier_offset_bytes = rex::align(barrier_offset_bytes, alignof(VkImageMemoryBarrier));
          image_memory_barriers = reinterpret_cast<const VkImageMemoryBarrier*>(
              reinterpret_cast<const uint8_t*>(stream) + barrier_offset_bytes);
          barrier_offset_bytes += sizeof(VkImageMemoryBarrier) * args.image_memory_barrier_count;
        }
        dfn.vkCmdPipelineBarrier(command_buffer, args.src_stage_mask, args.dst_stage_mask,
                                 args.dependency_flags, args.memory_barrier_count, memory_barriers,
                                 args.buffer_memory_barrier_count, buffer_memory_barriers,
                                 args.image_memory_barrier_count, image_memory_barriers);
      } break;

      case Command::kVkPushConstants: {
        auto& args = *reinterpret_cast<const ArgsVkPushConstants*>(stream);
        dfn.vkCmdPushConstants(
            command_buffer, args.layout, args.stage_flags, args.offset, args.size,
            reinterpret_cast<const uint8_t*>(stream) + sizeof(ArgsVkPushConstants));
      } break;

      case Command::kVrFrameCopy: {
        const auto& args = *reinterpret_cast<const ArgsVrFrameCopy*>(stream);
        ui::vulkan::vr::RecordFrameCopy(command_buffer, args.image, args.layout, args.width, args.height,
                                        args.screen, args.kind);
      } break;

      case Command::kVrEyeShift: {
        const auto& args = *reinterpret_cast<const ArgsVrEyeShift*>(stream);
        ui::vulkan::vr::RecordEyeShift(command_buffer, args.layout, args.perspective != 0, args.push, dfn);
      } break;

      case Command::kVrPushTextures: {
        const auto& args = *reinterpret_cast<const ArgsVrPushTextures*>(stream);
        ui::vulkan::vr::PushTextureDescriptors(
            command_buffer, args.layout, args.set, args.texture_count, args.sampler_count,
            reinterpret_cast<const VkDescriptorImageInfo*>(
                reinterpret_cast<const uint8_t*>(stream) +
                rex::align(sizeof(ArgsVrPushTextures), alignof(VkDescriptorImageInfo))));
      } break;

      case Command::kVrDlssDepth: {
        const auto& args = *reinterpret_cast<const ArgsVrDlssDepth*>(stream);
        ui::vulkan::vr::dlss::RecordDepthMotion(command_buffer, args.image, args.view, args.layout, args.width,
                                                args.height, args.view_projection, args.jitter_x, args.jitter_y, args.frame);
      } break;

      case Command::kVrMark: {
        if (vr_mark_hook) {  // both eyes: the categories sum them
          vr_mark_hook(command_buffer, *reinterpret_cast<const uint32_t*>(stream));
        }
      } break;

      case Command::kVkResetQueryPool: {
        auto& args = *reinterpret_cast<const ArgsVkResetQueryPool*>(stream);
        dfn.vkCmdResetQueryPool(command_buffer, args.query_pool, args.first_query,
                                args.query_count);
      } break;

      case Command::kVkSetBlendConstants: {
        auto& args = *reinterpret_cast<const ArgsVkSetBlendConstants*>(stream);
        dfn.vkCmdSetBlendConstants(command_buffer, args.blend_constants);
      } break;

      case Command::kVkSetDepthBias: {
        auto& args = *reinterpret_cast<const ArgsVkSetDepthBias*>(stream);
        dfn.vkCmdSetDepthBias(command_buffer, args.depth_bias_constant_factor,
                              args.depth_bias_clamp, args.depth_bias_slope_factor);
      } break;

      case Command::kVkSetScissor: {
        auto& args = *reinterpret_cast<const ArgsVkSetScissor*>(stream);
        dfn.vkCmdSetScissor(command_buffer, args.first_scissor, args.scissor_count,
                            reinterpret_cast<const VkRect2D*>(
                                reinterpret_cast<const uint8_t*>(stream) +
                                rex::align(sizeof(ArgsVkSetScissor), alignof(VkRect2D))));
      } break;

      case Command::kVkSetStencilCompareMask: {
        auto& args = *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilCompareMask(command_buffer, args.face_mask, args.mask_reference);
      } break;

      case Command::kVkSetStencilReference: {
        auto& args = *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilReference(command_buffer, args.face_mask, args.mask_reference);
      } break;

      case Command::kVkSetStencilWriteMask: {
        auto& args = *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilWriteMask(command_buffer, args.face_mask, args.mask_reference);
      } break;

      case Command::kVkSetViewport: {
        auto& args = *reinterpret_cast<const ArgsVkSetViewport*>(stream);
        dfn.vkCmdSetViewport(command_buffer, args.first_viewport, args.viewport_count,
                             reinterpret_cast<const VkViewport*>(
                                 reinterpret_cast<const uint8_t*>(stream) +
                                 rex::align(sizeof(ArgsVkSetViewport), alignof(VkViewport))));
      } break;

      default:
        assert_unhandled_case(header.command);
        break;
    }

    stream += header.arguments_size_elements;
    stream_remaining -= header.arguments_size_elements;
  }
}

void DeferredCommandBuffer::CmdVkPipelineBarrier(
    VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
    VkDependencyFlags dependency_flags, uint32_t memory_barrier_count,
    const VkMemoryBarrier* memory_barriers, uint32_t buffer_memory_barrier_count,
    const VkBufferMemoryBarrier* buffer_memory_barriers, uint32_t image_memory_barrier_count,
    const VkImageMemoryBarrier* image_memory_barriers) {
  size_t arguments_size = sizeof(ArgsVkPipelineBarrier);
  size_t memory_barriers_offset = 0;
  if (memory_barrier_count) {
    arguments_size = rex::align(arguments_size, alignof(VkMemoryBarrier));
    memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkMemoryBarrier) * memory_barrier_count;
  }
  size_t buffer_memory_barriers_offset = 0;
  if (buffer_memory_barrier_count) {
    arguments_size = rex::align(arguments_size, alignof(VkBufferMemoryBarrier));
    buffer_memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkBufferMemoryBarrier) * buffer_memory_barrier_count;
  }
  size_t image_memory_barriers_offset = 0;
  if (image_memory_barrier_count) {
    arguments_size = rex::align(arguments_size, alignof(VkImageMemoryBarrier));
    image_memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkImageMemoryBarrier) * image_memory_barrier_count;
  }
  uint8_t* args_ptr =
      reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkPipelineBarrier, arguments_size));
  auto& args = *reinterpret_cast<ArgsVkPipelineBarrier*>(args_ptr);
  args.src_stage_mask = src_stage_mask;
  args.dst_stage_mask = dst_stage_mask;
  args.dependency_flags = dependency_flags;
  args.memory_barrier_count = memory_barrier_count;
  args.buffer_memory_barrier_count = buffer_memory_barrier_count;
  args.image_memory_barrier_count = image_memory_barrier_count;
  if (memory_barrier_count) {
    std::memcpy(args_ptr + memory_barriers_offset, memory_barriers,
                sizeof(VkMemoryBarrier) * memory_barrier_count);
  }
  if (buffer_memory_barrier_count) {
    std::memcpy(args_ptr + buffer_memory_barriers_offset, buffer_memory_barriers,
                sizeof(VkBufferMemoryBarrier) * buffer_memory_barrier_count);
  }
  if (image_memory_barrier_count) {
    std::memcpy(args_ptr + image_memory_barriers_offset, image_memory_barriers,
                sizeof(VkImageMemoryBarrier) * image_memory_barrier_count);
  }
}

void DeferredCommandBuffer::CmdVkBeginRendering(const VkRenderingInfo* rendering_info) {
  assert_null(rendering_info->pNext);

  size_t arguments_size =
      rex::align(sizeof(ArgsVkBeginRendering), alignof(VkRenderingAttachmentInfo));
  size_t color_attachments_offset = arguments_size;
  arguments_size += sizeof(VkRenderingAttachmentInfo) * rendering_info->colorAttachmentCount;
  size_t depth_attachment_offset = arguments_size;
  if (rendering_info->pDepthAttachment) {
    arguments_size += sizeof(VkRenderingAttachmentInfo);
  }
  size_t stencil_attachment_offset = arguments_size;
  if (rendering_info->pStencilAttachment) {
    arguments_size += sizeof(VkRenderingAttachmentInfo);
  }

  uint8_t* args_ptr =
      reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkBeginRendering, arguments_size));
  auto& args = *reinterpret_cast<ArgsVkBeginRendering*>(args_ptr);
  args.flags = rendering_info->flags;
  args.render_area = rendering_info->renderArea;
  args.layer_count = rendering_info->layerCount;
  args.view_mask = rendering_info->viewMask;
  args.color_attachment_count = rendering_info->colorAttachmentCount;
  args.has_depth_attachment = rendering_info->pDepthAttachment != nullptr;
  args.has_stencil_attachment = rendering_info->pStencilAttachment != nullptr;

  if (rendering_info->colorAttachmentCount) {
    std::memcpy(args_ptr + color_attachments_offset, rendering_info->pColorAttachments,
                sizeof(VkRenderingAttachmentInfo) * rendering_info->colorAttachmentCount);
  }
  if (rendering_info->pDepthAttachment) {
    std::memcpy(args_ptr + depth_attachment_offset, rendering_info->pDepthAttachment,
                sizeof(VkRenderingAttachmentInfo));
  }
  if (rendering_info->pStencilAttachment) {
    std::memcpy(args_ptr + stencil_attachment_offset, rendering_info->pStencilAttachment,
                sizeof(VkRenderingAttachmentInfo));
  }
}

void* DeferredCommandBuffer::WriteCommand(Command command, size_t arguments_size_bytes) {
  size_t arguments_size_elements =
      (arguments_size_bytes + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);
  std::vector<uintmax_t>& stream = prologue_ ? prologue_stream_ : command_stream_;  // VR fork: or the prologue
  size_t offset = stream.size();
  stream.resize(offset + kCommandHeaderSizeElements + arguments_size_elements);
  CommandHeader& header = *reinterpret_cast<CommandHeader*>(stream.data() + offset);
  header.command = command;
  header.arguments_size_elements = uint32_t(arguments_size_elements);
  return stream.data() + (offset + kCommandHeaderSizeElements);
}

}  // namespace rex::graphics::vulkan
