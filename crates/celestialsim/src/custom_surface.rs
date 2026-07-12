//! Custom **GPU** surface layers: splice a user's terrain GLSL into the library
//! template and drive the per-slot surface buffers on the render device.
//!
//! This is the code-authored terrain extension point for consumers who want
//! procedural terrain that runs entirely on the GPU (no CPU bake pool). The user
//! writes a small `.glsl` file defining `terrain_height` / `terrain_color` (and
//! optionally `terrain_normal`); [`assemble_source`] wraps it in
//! [`TEMPLATE`](self) — which owns the per-texel gnomonic direction mapping, the
//! finite-difference auto-normal, and the writes into the same
//! `surface_color/height/normal` buffers the built-in realize/bake shaders read
//! when `surface_enabled == 1`. Godot's `RenderingDevice` compiles the assembled
//! GLSL at runtime, so **no `slangc` and nothing new ships** — it reuses Godot's
//! own shader compiler. The compiled pipeline runs as the
//! `celestial/chunk-surface-custom` node (`crate::chunk_nodes::surface_custom`),
//! between upload and realize.
//!
//! Contrast with [`crate::surface::CpuSurfaceProvider`], which bakes the same
//! buffers on CPU worker threads (for data-driven / streaming terrain).

/// The library GLSL template. The user's source replaces the
/// `// __CELS_USER_CODE__` marker line.
pub const TEMPLATE: &str = include_str!("../shaders/custom_surface.glsl");

/// The marker line in [`TEMPLATE`] where user code is spliced in.
const USER_CODE_MARKER: &str = "// __CELS_USER_CODE__";

/// The marker line in [`TEMPLATE`] where per-param `#define`s are spliced
/// (before the user code, so the user's functions can reference them).
const USER_DEFINES_MARKER: &str = "// __CELS_USER_DEFINES__";

/// Max generic user params (`@export var name: float`) surfaced to a GPU
/// builder's `.glsl`. Matches the `float cels_user[16]` tail in the template's
/// binding-4 `Params` block and the pad in [`pack_params`].
pub const MAX_USER_PARAMS: usize = 16;

/// Errors assembling a custom-surface shader from user GLSL.
#[derive(Debug, PartialEq, Eq)]
pub enum AssembleError {
    /// The user source does not define a required function.
    MissingFn(&'static str),
    /// The template lost its user-code marker (a library bug, not user error).
    TemplateMarkerMissing,
}

impl std::fmt::Display for AssembleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssembleError::MissingFn(name) => write!(
                f,
                "custom-surface GLSL must define `{name}` (see docs/custom_terrain_gpu.md)"
            ),
            AssembleError::TemplateMarkerMissing => {
                write!(f, "custom-surface template is missing its user-code marker")
            }
        }
    }
}

/// Splice `user_glsl` into [`TEMPLATE`], producing a complete compute shader.
///
/// Fails fast (before touching the GPU) if the user omitted a required function
/// so the error names the missing symbol instead of surfacing as an opaque GLSL
/// compile error. `terrain_normal` is optional and not checked.
pub fn assemble_source(user_glsl: &str) -> Result<String, AssembleError> {
    assemble_source_with_params(user_glsl, &[])
}

/// Like [`assemble_source`], but also splices a `#define` block — one line per
/// `param_names` entry — at the defines marker so the user's `.glsl` can
/// reference each `@export var name: float` on the builder as the UPPERCASE
/// `NAME` (backed by `P.cels_user[<index>]`). Names past [`MAX_USER_PARAMS`] are
/// ignored (they have no backing slot).
pub fn assemble_source_with_params(
    user_glsl: &str,
    param_names: &[String],
) -> Result<String, AssembleError> {
    for required in ["terrain_height", "terrain_color"] {
        if !user_glsl.contains(required) {
            return Err(AssembleError::MissingFn(match required {
                "terrain_height" => "terrain_height",
                _ => "terrain_color",
            }));
        }
    }
    // Each marker MUST appear exactly once: `str::replace` is global, so a stray
    // mention elsewhere (e.g. in a header comment) would also be replaced and
    // inject content — with newlines — mid-comment, breaking the shader.
    if TEMPLATE.matches(USER_CODE_MARKER).count() != 1
        || TEMPLATE.matches(USER_DEFINES_MARKER).count() != 1
    {
        return Err(AssembleError::TemplateMarkerMissing);
    }
    let mut defines = String::new();
    for (i, name) in param_names.iter().take(MAX_USER_PARAMS).enumerate() {
        defines.push_str(&format!("#define {} (P.cels_user[{}])\n", name.to_uppercase(), i));
    }
    // Splice defines first, then user code (order independent — distinct markers).
    let with_defines = TEMPLATE.replace(USER_DEFINES_MARKER, defines.trim_end());
    Ok(with_defines.replace(USER_CODE_MARKER, user_glsl))
}

/// Pack the custom-surface params buffer (std430) read by the template's
/// binding-4 `Params` block: the four fixed fields
/// `{chunk_count, tile_res, water_height, height_scale}` followed by
/// [`MAX_USER_PARAMS`] user floats (`cels_user[16]`). Extra `user` values are
/// ignored; missing ones are zero-filled. Total = `16 + 4*MAX_USER_PARAMS` bytes.
pub fn pack_params(
    chunk_count: u32,
    tile_res: u32,
    water_height: f32,
    height_scale: f32,
    user: &[f32],
) -> Vec<u8> {
    let mut out = vec![0u8; 16 + 4 * MAX_USER_PARAMS];
    out[0..4].copy_from_slice(&chunk_count.to_le_bytes());
    out[4..8].copy_from_slice(&tile_res.to_le_bytes());
    out[8..12].copy_from_slice(&water_height.to_le_bytes());
    out[12..16].copy_from_slice(&height_scale.to_le_bytes());
    for (i, v) in user.iter().take(MAX_USER_PARAMS).enumerate() {
        let off = 16 + i * 4;
        out[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const OK_USER: &str = r#"
        float terrain_height(vec3 dir) { return 0.05 * sin(dir.x * 8.0); }
        vec3 terrain_color(vec3 dir, float h) { return vec3(h, 0.5, 0.2); }
    "#;

    #[test]
    fn assembles_user_code_into_template() {
        let src = assemble_source(OK_USER).expect("valid user code assembles");
        // The user body is present and the marker is gone.
        assert!(src.contains("0.05 * sin(dir.x * 8.0)"));
        assert!(!src.contains(USER_CODE_MARKER));
        // Template scaffolding survives (entry point + one of the buffers).
        assert!(src.contains("void main()"));
        assert!(src.contains("surface_height["));
    }

    #[test]
    fn missing_height_fn_is_rejected_before_gpu() {
        let bad = "vec3 terrain_color(vec3 dir, float h) { return vec3(0.0); }";
        assert_eq!(assemble_source(bad), Err(AssembleError::MissingFn("terrain_height")));
    }

    #[test]
    fn missing_color_fn_is_rejected_before_gpu() {
        let bad = "float terrain_height(vec3 dir) { return 0.0; }";
        assert_eq!(assemble_source(bad), Err(AssembleError::MissingFn("terrain_color")));
    }

    #[test]
    fn template_has_exactly_one_marker() {
        // Exactly one: zero => no splice point; two+ => `replace` injects user
        // code (with newlines) into a comment mention and corrupts the shader
        // (the real bug this guards). A multi-line user body must land ONLY at
        // the true marker.
        assert_eq!(TEMPLATE.matches(USER_CODE_MARKER).count(), 1);
    }

    #[test]
    fn multiline_user_code_assembles_without_corrupting_template() {
        // The header must not carry a second marker mention: a real multi-line
        // file (comments + several statements) must splice cleanly.
        let multiline = "// a comment\nfloat terrain_height(vec3 dir) {\n  return 0.1;\n}\nvec3 terrain_color(vec3 dir, float h) {\n  return vec3(h);\n}\n";
        let src = assemble_source(multiline).expect("multiline assembles");
        // The template header text after the marker survives intact (proof the
        // splice happened at the true marker, not inside the header comment).
        assert!(src.contains("void main()"));
        assert!(src.contains("cels_pack_rgba8"));
    }

    #[test]
    fn pack_params_layout_fixed_head_plus_user_tail() {
        let b = pack_params(3, 256, 0.45, 0.18, &[1.5, -2.0]);
        assert_eq!(b.len(), 16 + 4 * MAX_USER_PARAMS);
        assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), 3);
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]), 256);
        assert_eq!(f32::from_le_bytes([b[8], b[9], b[10], b[11]]), 0.45);
        assert_eq!(f32::from_le_bytes([b[12], b[13], b[14], b[15]]), 0.18);
        // First two user floats land in the tail; the rest are zero-filled.
        assert_eq!(f32::from_le_bytes([b[16], b[17], b[18], b[19]]), 1.5);
        assert_eq!(f32::from_le_bytes([b[20], b[21], b[22], b[23]]), -2.0);
        assert_eq!(f32::from_le_bytes([b[24], b[25], b[26], b[27]]), 0.0);
    }

    #[test]
    fn pack_params_ignores_user_overflow() {
        let many: Vec<f32> = (0..MAX_USER_PARAMS + 4).map(|i| i as f32).collect();
        let b = pack_params(0, 0, 0.0, 0.0, &many);
        assert_eq!(b.len(), 16 + 4 * MAX_USER_PARAMS);
        // Last in-range slot is index MAX_USER_PARAMS-1.
        let off = 16 + (MAX_USER_PARAMS - 1) * 4;
        assert_eq!(
            f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]),
            (MAX_USER_PARAMS - 1) as f32
        );
    }

    #[test]
    fn param_name_produces_define_at_expected_index() {
        let src = assemble_source_with_params(OK_USER, &["ridge_sharpness".into(), "snow_line".into()])
            .expect("assembles with params");
        assert!(src.contains("#define RIDGE_SHARPNESS (P.cels_user[0])"));
        assert!(src.contains("#define SNOW_LINE (P.cels_user[1])"));
        // Markers are gone and the user body survives.
        assert!(!src.contains(USER_DEFINES_MARKER));
        assert!(!src.contains(USER_CODE_MARKER));
        assert!(src.contains("void main()"));
    }

    #[test]
    fn template_has_exactly_one_defines_marker() {
        assert_eq!(TEMPLATE.matches(USER_DEFINES_MARKER).count(), 1);
    }
}
