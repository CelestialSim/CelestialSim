use godot::prelude::*;

#[allow(dead_code)]
mod buffer_info;
#[allow(dead_code)]
mod compute_utils;

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
