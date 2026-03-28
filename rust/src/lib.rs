use godot::prelude::*;

#[allow(dead_code)]
mod algo;
#[allow(dead_code)]
mod buffer_info;
mod celestial;
mod compositor;
#[allow(dead_code)]
mod compute_utils;
#[allow(dead_code)]
mod cpu_subdivide;
mod debug_dump;
#[allow(dead_code)]
mod initial_state;
mod layer_resources;
#[allow(dead_code)]
mod layers;
#[allow(dead_code)]
mod state;
#[allow(dead_code)]
mod texture_gen;

struct CelestialSimExtension;

#[gdextension]
unsafe impl ExtensionLibrary for CelestialSimExtension {}

/// A simple test node to verify the Rust GDExtension loads correctly in Godot.
#[derive(GodotClass)]
#[class(base = Node)]
struct CesRustTest {
    base: Base<Node>,
}

#[godot_api]
impl INode for CesRustTest {
    fn init(base: Base<Node>) -> Self {
        Self { base }
    }

    fn ready(&mut self) {
        godot_print!("[CelestialSimRust] Hello from Rust GDExtension!");
    }
}
