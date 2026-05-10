use godot::builtin::PackedByteArray;
use godot::classes::rendering_device::{
    DataFormat, DriverResource, TextureSamples, TextureType, TextureUsageBits,
};
use godot::classes::{
    RdTextureFormat, RdTextureView, RenderingDevice, RenderingServer, Texture2Drd,
};
use godot::obj::{Gd, NewGd, Singleton};
use godot::prelude::Rid;

use crate::compute_utils::{free_rid_on_render_thread, on_render_thread_sync, RdSend};

pub const POSITION_TEXTURE_FORMAT: DataFormat = DataFormat::R32G32B32A32_SFLOAT;
const RETIRED_TEXTURE_PAIR_GRACE_LIMIT: usize = 2;

pub fn shared_texture_2d_usage_bits() -> TextureUsageBits {
    TextureUsageBits::SAMPLING_BIT
        | TextureUsageBits::STORAGE_BIT
        | TextureUsageBits::CAN_UPDATE_BIT
        | TextureUsageBits::CAN_COPY_FROM_BIT
}

pub fn create_shared_2d_texture_rids(
    local_rd: &mut Gd<RenderingDevice>,
    width: u32,
    height: u32,
    format: DataFormat,
    usage_bits: TextureUsageBits,
) -> (Rid, Rid) {
    let driver_handle = on_render_thread_sync(move || {
        let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();

        let mut tex_format = RdTextureFormat::new_gd();
        tex_format.set_texture_type(TextureType::TYPE_2D);
        tex_format.set_format(format);
        tex_format.set_width(width);
        tex_format.set_height(height);
        tex_format.set_depth(1);
        tex_format.set_array_layers(1);
        tex_format.set_mipmaps(1);
        tex_format.set_samples(TextureSamples::SAMPLES_1);
        tex_format.set_usage_bits(usage_bits);

        let view = RdTextureView::new_gd();
        let main_rid = main_rd.texture_create(&tex_format, &view);
        assert!(
            main_rid.is_valid(),
            "Failed to create shared main 2D texture"
        );

        let handle = main_rd.get_driver_resource(DriverResource::TEXTURE, main_rid, 0);
        (main_rid, handle)
    });

    let (main_rid, handle) = driver_handle;

    let local_rd_send = RdSend(local_rd.clone());
    let local_rid = on_render_thread_sync(move || {
        let mut rd = local_rd_send;
        rd.0.texture_create_from_extension(
            TextureType::TYPE_2D,
            format,
            TextureSamples::SAMPLES_1,
            usage_bits,
            handle,
            width as u64,
            height as u64,
            1,
            1,
        )
    });

    if !local_rid.is_valid() {
        free_main_rid_sync(main_rid);
        panic!("Failed to create local 2D texture from extension");
    }

    (main_rid, local_rid)
}

pub fn wrap_main_texture_2d(main_rid: Rid) -> Option<Gd<Texture2Drd>> {
    if !main_rid.is_valid() {
        return None;
    }
    let mut tex = Texture2Drd::new_gd();
    tex.set_texture_rd_rid(main_rid);
    Some(tex)
}

pub fn update_shared_2d_texture_rgba32f(
    local_rd: &mut Gd<RenderingDevice>,
    local_rid: Rid,
    texels: &[f32],
) {
    if !local_rid.is_valid() {
        return;
    }

    let rd_send = RdSend(local_rd.clone());
    let bytes = bytemuck::cast_slice(texels).to_vec();
    on_render_thread_sync(move || {
        let mut rd = rd_send;
        let packed = PackedByteArray::from(bytes.as_slice());
        let _ = rd.0.texture_update(local_rid, 0, &packed);
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedTexture2DExtent {
    pub width: u32,
    pub height: u32,
}

impl PackedTexture2DExtent {
    pub fn capacity(self) -> u32 {
        self.width.saturating_mul(self.height)
    }
}

pub fn packed_1d_to_2d_extent(element_count: u32) -> PackedTexture2DExtent {
    let count = element_count.max(1);
    let width = (count as f64).sqrt().ceil() as u32;
    let width = width.max(1);
    let height = count.div_ceil(width).max(1);
    PackedTexture2DExtent { width, height }
}

#[derive(Clone, Copy, Debug)]
struct SharedTexture2DRids {
    main: Rid,
    local: Rid,
}

impl Default for SharedTexture2DRids {
    fn default() -> Self {
        Self {
            main: Rid::Invalid,
            local: Rid::Invalid,
        }
    }
}

impl SharedTexture2DRids {
    fn new(main: Rid, local: Rid) -> Self {
        Self { main, local }
    }

    fn is_valid(&self) -> bool {
        self.main.is_valid() && self.local.is_valid()
    }
}

fn free_main_rid_sync(rid: Rid) {
    if !rid.is_valid() {
        return;
    }
    on_render_thread_sync(move || {
        let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();
        main_rd.free_rid(rid);
    });
}

pub struct SharedPositionTexture {
    front: SharedTexture2DRids,
    back: SharedTexture2DRids,
    retired: Vec<SharedTexture2DRids>,
    extent: PackedTexture2DExtent,
    capacity_vertices: u32,
}

impl SharedPositionTexture {
    pub fn new() -> Self {
        Self {
            front: SharedTexture2DRids::default(),
            back: SharedTexture2DRids::default(),
            retired: Vec::new(),
            extent: PackedTexture2DExtent {
                width: 0,
                height: 0,
            },
            capacity_vertices: 0,
        }
    }

    pub fn has_textures(&self) -> bool {
        self.front.is_valid() && self.back.is_valid()
    }

    pub fn capacity_vertices(&self) -> u32 {
        self.capacity_vertices
    }

    pub fn extent(&self) -> PackedTexture2DExtent {
        self.extent
    }

    pub fn front_main_rid(&self) -> Rid {
        self.front.main
    }

    pub fn front_local_rid(&self) -> Rid {
        self.front.local
    }

    pub fn back_main_rid(&self) -> Rid {
        self.back.main
    }

    pub fn back_local_rid(&self) -> Rid {
        self.back.local
    }

    pub fn front_main_texture(&self) -> Option<Gd<Texture2Drd>> {
        wrap_main_texture_2d(self.front.main)
    }

    pub fn ensure_capacity(&mut self, local_rd: &mut Gd<RenderingDevice>, required_vertices: u32) {
        if self.has_textures() && self.capacity_vertices >= required_vertices.max(1) {
            return;
        }

        self.retire_current_textures(local_rd);

        let extent = packed_1d_to_2d_extent(required_vertices);
        let usage_bits = shared_texture_2d_usage_bits();
        let front = create_shared_2d_texture_rids(
            local_rd,
            extent.width,
            extent.height,
            POSITION_TEXTURE_FORMAT,
            usage_bits,
        );
        let back = create_shared_2d_texture_rids(
            local_rd,
            extent.width,
            extent.height,
            POSITION_TEXTURE_FORMAT,
            usage_bits,
        );

        self.front = SharedTexture2DRids::new(front.0, front.1);
        self.back = SharedTexture2DRids::new(back.0, back.1);
        self.extent = extent;
        self.capacity_vertices = extent.capacity();
    }

    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.front, &mut self.back);
    }

    fn retire_current_textures(&mut self, local_rd: &mut Gd<RenderingDevice>) {
        let to_free = self.rotate_retired_generation();
        for retired in to_free {
            Self::free_texture_pair_local(local_rd, retired);
        }
    }

    fn rotate_retired_generation(&mut self) -> Vec<SharedTexture2DRids> {
        let to_free = std::mem::take(&mut self.retired);
        if self.front.is_valid() {
            self.retired.push(self.front);
        }
        if self.back.is_valid() {
            self.retired.push(self.back);
        }
        debug_assert!(
            self.retired.len() <= RETIRED_TEXTURE_PAIR_GRACE_LIMIT,
            "shared position texture should retain at most one prior front/back generation"
        );
        self.front = SharedTexture2DRids::default();
        self.back = SharedTexture2DRids::default();
        to_free
    }

    fn free_texture_pair_local(local_rd: &mut Gd<RenderingDevice>, rids: SharedTexture2DRids) {
        if rids.local.is_valid() {
            free_rid_on_render_thread(local_rd, rids.local);
        }
        free_main_rid_sync(rids.main);
    }

    pub fn dispose_local(&mut self, local_rd: &mut Gd<RenderingDevice>) {
        Self::free_texture_pair_local(local_rd, self.front);
        Self::free_texture_pair_local(local_rd, self.back);
        for retired in self.retired.drain(..) {
            Self::free_texture_pair_local(local_rd, retired);
        }

        self.front = SharedTexture2DRids::default();
        self.back = SharedTexture2DRids::default();
        self.extent = PackedTexture2DExtent {
            width: 0,
            height: 0,
        };
        self.capacity_vertices = 0;
    }

    pub fn dispose_direct(
        &mut self,
        main_rd: &mut Gd<RenderingDevice>,
        local_rd: &mut Gd<RenderingDevice>,
    ) {
        Self::free_texture_pair_direct(main_rd, local_rd, self.front);
        Self::free_texture_pair_direct(main_rd, local_rd, self.back);
        for retired in self.retired.drain(..) {
            Self::free_texture_pair_direct(main_rd, local_rd, retired);
        }

        self.front = SharedTexture2DRids::default();
        self.back = SharedTexture2DRids::default();
        self.extent = PackedTexture2DExtent {
            width: 0,
            height: 0,
        };
        self.capacity_vertices = 0;
    }

    fn free_texture_pair_direct(
        main_rd: &mut Gd<RenderingDevice>,
        local_rd: &mut Gd<RenderingDevice>,
        rids: SharedTexture2DRids,
    ) {
        if rids.local.is_valid() {
            free_rid_on_render_thread(local_rd, rids.local);
        }
        if rids.main.is_valid() {
            free_rid_on_render_thread(main_rd, rids.main);
        }
    }
}

impl Default for SharedPositionTexture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_1d_to_2d_extent_zero_uses_one_texel() {
        let extent = packed_1d_to_2d_extent(0);
        assert_eq!(extent.width, 1);
        assert_eq!(extent.height, 1);
        assert_eq!(extent.capacity(), 1);
    }

    #[test]
    fn packed_1d_to_2d_extent_capacity_covers_request() {
        for count in [1u32, 2, 3, 4, 5, 17, 1024, 109220, 245760] {
            let extent = packed_1d_to_2d_extent(count);
            assert!(extent.capacity() >= count);
            assert!(extent.width > 0);
            assert!(extent.height > 0);
        }
    }

    #[test]
    fn packed_1d_to_2d_extent_is_nearly_square() {
        for count in [1u32, 2, 7, 64, 65, 1000, 109220] {
            let extent = packed_1d_to_2d_extent(count);
            assert!(extent.width >= extent.height);
            assert!(extent.width - extent.height <= 1);
        }
    }

    #[test]
    fn shared_position_texture_new_starts_empty() {
        let tex = SharedPositionTexture::new();
        assert!(!tex.has_textures());
        assert_eq!(tex.capacity_vertices(), 0);
        assert_eq!(tex.front_main_rid(), Rid::Invalid);
        assert_eq!(tex.back_local_rid(), Rid::Invalid);
        assert!(tex.retired.is_empty());
    }

    #[test]
    fn rotate_retired_generation_replaces_previous_generation_instead_of_accumulating() {
        let mut tex = SharedPositionTexture::new();
        tex.retired = vec![
            SharedTexture2DRids::default(),
            SharedTexture2DRids::default(),
        ];

        let freed = tex.rotate_retired_generation();

        assert_eq!(freed.len(), 2);
        assert!(tex.retired.is_empty());
        assert!(tex.retired.len() <= RETIRED_TEXTURE_PAIR_GRACE_LIMIT);
    }
}
