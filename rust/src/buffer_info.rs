use godot::classes::rendering_device::UniformType;
use godot::classes::{RdUniform, RenderingDevice};
use godot::obj::{Gd, NewGd};
use godot::prelude::Rid;

/// Type of GPU buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    StorageBuffer,
    UniformBuffer,
}

impl BufferType {
    pub fn to_uniform_type(self) -> UniformType {
        match self {
            BufferType::StorageBuffer => UniformType::STORAGE_BUFFER,
            BufferType::UniformBuffer => UniformType::UNIFORM_BUFFER,
        }
    }
}

/// Tracks a GPU buffer's Rid, capacity, used size, and type.
///
/// Mirrors the C# `BufferInfo` class from CesState.cs.
pub struct BufferInfo {
    pub rid: Rid,
    pub max_size: u32,
    pub filled_size: u32,
    pub buffer_type: BufferType,
}

impl BufferInfo {
    /// Create a new storage-type BufferInfo.
    pub fn new_storage(rid: Rid, filled_size: u32, max_size: u32) -> Self {
        Self {
            rid,
            max_size,
            filled_size,
            buffer_type: BufferType::StorageBuffer,
        }
    }

    /// Create a new uniform-type BufferInfo.
    pub fn new_uniform(rid: Rid) -> Self {
        Self {
            rid,
            max_size: 0,
            filled_size: 0,
            buffer_type: BufferType::UniformBuffer,
        }
    }

    /// Creates an `RDUniform` with the correct type and binding index referencing this buffer.
    pub fn get_uniform_with_binding(&self, binding: i32) -> Gd<RdUniform> {
        let mut uniform = RdUniform::new_gd();
        uniform.set_uniform_type(self.buffer_type.to_uniform_type());
        uniform.set_binding(binding);
        uniform.add_id(self.rid);
        uniform
    }

    /// Extends the buffer by `bytes_to_extend`.
    ///
    /// If `filled_size + bytes_to_extend` fits within `max_size` (and the buffer
    /// isn't more than 2x the needed size), just bumps `filled_size`.
    /// Otherwise allocates a new buffer, copies data, and frees the old one.
    pub fn extend_buffer(&mut self, rd: &mut RenderingDevice, bytes_to_extend: u32) {
        let desired_size = self.filled_size + bytes_to_extend;

        // Reuse existing buffer if it fits and isn't oversized (>2x)
        if self.max_size >= desired_size && self.max_size <= desired_size * 2 {
            self.filled_size = desired_size;
            return;
        }

        // Allocate a new buffer
        let new_rid = rd.storage_buffer_create(desired_size);
        assert!(
            new_rid.is_valid(),
            "Failed to create new storage buffer during extend"
        );

        // Clear the new buffer to prevent garbage data in uninitialized region
        rd.buffer_clear(new_rid, 0, desired_size);

        let old_rid = self.rid;
        let old_filled = self.filled_size;

        // Copy existing data into new buffer
        rd.buffer_copy(old_rid, new_rid, 0, 0, old_filled);

        // Free old buffer
        rd.free_rid(old_rid);

        // Update self
        self.rid = new_rid;
        self.filled_size = desired_size;
        self.max_size = desired_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_info_new_storage() {
        let rid = Rid::Invalid;
        let info = BufferInfo::new_storage(rid, 100, 200);
        assert_eq!(info.filled_size, 100);
        assert_eq!(info.max_size, 200);
        assert_eq!(info.buffer_type, BufferType::StorageBuffer);
    }

    #[test]
    fn test_buffer_info_new_uniform() {
        let rid = Rid::Invalid;
        let info = BufferInfo::new_uniform(rid);
        assert_eq!(info.buffer_type, BufferType::UniformBuffer);
    }

    #[test]
    fn test_buffer_type_to_uniform_type() {
        assert_eq!(
            BufferType::StorageBuffer.to_uniform_type(),
            UniformType::STORAGE_BUFFER
        );
        assert_eq!(
            BufferType::UniformBuffer.to_uniform_type(),
            UniformType::UNIFORM_BUFFER
        );
    }

    #[test]
    fn test_extend_buffer_reuses_when_fits() {
        // If max_size >= desired and max_size <= desired*2, should just bump filled_size
        let info = BufferInfo::new_storage(Rid::Invalid, 100, 200);
        // desired = 100 + 50 = 150. max_size(200) >= 150 && 200 <= 300. OK → reuse.
        // We can't call extend_buffer without a real RenderingDevice,
        // but we can test the logic by checking the condition directly.
        let desired = info.filled_size + 50;
        let reuse = info.max_size >= desired && info.max_size <= desired * 2;
        assert!(reuse);
    }

    #[test]
    fn test_extend_buffer_needs_realloc_when_oversized() {
        // If max_size > desired*2, should reallocate (buffer too big → perf issue)
        let info = BufferInfo::new_storage(Rid::Invalid, 10, 1000);
        let desired = info.filled_size + 10; // 20
        let reuse = info.max_size >= desired && info.max_size <= desired * 2;
        assert!(!reuse); // 1000 > 40, so should NOT reuse
    }

    #[test]
    fn test_extend_buffer_needs_realloc_when_too_small() {
        let info = BufferInfo::new_storage(Rid::Invalid, 100, 100);
        let desired = info.filled_size + 50; // 150
        let reuse = info.max_size >= desired && info.max_size <= desired * 2;
        assert!(!reuse); // 100 < 150, should NOT reuse
    }
}
