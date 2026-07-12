// Surface shader for GPU-realized quadtree chunks (Phase 2, CEL-62).
//
// Each MultiMesh instance is one chunk: a res*res triangular grid (the
// `reference_chunk_mesh` template). The realize compute pass (ChunkRealize.slang)
// wrote every UNIQUE chunk vertex once into two GPU-written textures:
//   pos_tex   — 1 rgba32f texel / vertex: xyz = world position (sampled for VERTEX)
//   verts_tex — 2 rgba16f texels / vertex: (color+layer), (normal+height)
// The instance carries its vertex-pool SLOT in INSTANCE_CUSTOM.r; the local
// vertex index is VERTEX_ID, so the global pool index is slot*verts_per_chunk +
// VERTEX_ID. A spatial shader can only sample textures (not the storage buffer
// the readback test reads), which is why pos_tex exists.
//
// Backface culling: the reference chunk mesh is uniformly CCW-wound (chunk_mesh.rs),
// so back-facing terrain (far side of the planet, and away-facing chunks) is culled
// — roughly halving rasterized triangles vs the old cull_disabled.

shader_type spatial;
render_mode cull_back, world_vertex_coords;

uniform sampler2D pos_tex : filter_nearest;
uniform sampler2D verts_tex : filter_nearest;
// Phase 4: per-chunk colour + world-normal detail atlases. `filter_nearest`
// avoids cross-slot bleed at tile/atlas-row boundaries.
uniform sampler2D color_atlas : filter_nearest;
uniform sampler2D normal_atlas : filter_nearest;
uniform int attr_w = 4096;
uniform int verts_per_chunk = 153;
uniform int tile_res = 32;
// Phase 5 geomorph: edge resolution of the chunk grid. The vertex shader uses it
// to decode each vertex's (i, j) lattice coords and snap to the even (parent)
// sublattice for the coarse morph target.
uniform int chunk_res = 16;
// Tint each chunk by its slot so the chunk tiling is visible (verification aid).
uniform bool lod_colors = false;

varying flat int v_slot;
varying vec2 v_uv; // chunk-local barycentric (wb, wc) interpolated per pixel

// Distinct hue per integer key (same 0.137 hue spin as terrain_surface's palette).
vec3 hue_color(int key) {
	float hue = fract(float(key) * 0.137);
	float h6 = hue * 6.0;
	float c = 0.95 * 0.8;
	float x = c * (1.0 - abs(mod(h6, 2.0) - 1.0));
	float m = 0.95 - c;
	vec3 rgb;
	if (h6 < 1.0) { rgb = vec3(c, x, 0.0); }
	else if (h6 < 2.0) { rgb = vec3(x, c, 0.0); }
	else if (h6 < 3.0) { rgb = vec3(0.0, c, x); }
	else if (h6 < 4.0) { rgb = vec3(0.0, x, c); }
	else if (h6 < 5.0) { rgb = vec3(x, 0.0, c); }
	else { rgb = vec3(c, 0.0, x); }
	return rgb + m;
}

// Realized world position of local vertex `localL` in this chunk's pool slot.
vec3 pool_pos(int slot, int localL) {
	int g = slot * verts_per_chunk + localL;
	return texelFetch(pos_tex, ivec2(g % attr_w, g / attr_w), 0).xyz;
}

// Canonical interior local index for lattice (i, j): L = i*(2*res+3-i)/2 + j.
// i*(2*res+3-i) is even, so the integer /2 is exact.
int interior_L(int i, int j) {
	return (i * (2 * chunk_res + 3 - i)) / 2 + j;
}

// Phase 5 geomorph. The chunk "pops" when the quadtree subdivides/merges because
// vertices jump between LOD levels. To smooth that, each instance carries a morph
// factor (INSTANCE_CUSTOM.g, float index 13: 1 = full detail, 0 = coarse/parent
// resolution) that ramps with camera distance (computed CPU-side per chunk). Each
// vertex blends between its FINE realized position and its COARSE morph target —
// the position it would have on the parent (even-resolution) grid. A vertex shared
// with the coarse grid (both lattice coords even) keeps its own position; an
// edge-midpoint vertex (a coord is odd) morphs to the MIDPOINT of the two even
// vertices of the edge it bisects (the straight coarse edge, fine displacement
// averaged out). Averaging the two endpoints — NOT snapping to one nearest even
// vertex, which would collapse midpoints onto corners and bunch the mesh into
// blobs — is what makes the fade geometrically correct. This rides in the per-frame
// instance buffer ONLY (not the cached realize/atlas), so a moving camera
// re-uploads just the instance buffer.
void vertex() {
	int slot = int(INSTANCE_CUSTOM.r);
	float morph = INSTANCE_CUSTOM.g;
	int interior_count = (chunk_res + 1) * (chunk_res + 2) / 2;

	// FINE: this vertex's GPU-realized world position.
	int gv = slot * verts_per_chunk + VERTEX_ID;
	vec3 fine = texelFetch(pos_tex, ivec2(gv % attr_w, gv / attr_w), 0).xyz;

	// COARSE morph target: where this vertex sits on the parent (even-res) grid.
	vec3 coarse;
	if (VERTEX_ID < interior_count) {
		// Interior vertex: recover (i, j) from the chunk-local bary UV (u = wb =
		// i/res, v = wc = j/res). i, j are exact integers (UV is the vertex attr).
		int i = int(round(UV.x * float(chunk_res)));
		int j = int(round(UV.y * float(chunk_res)));
		int ip = i % 2;
		int jp = j % 2;
		if (ip == 0 && jp == 0) {
			coarse = fine; // shared with the coarse grid — no offset
		} else if (ip == 1 && jp == 0) {
			// Midpoint of the horizontal coarse edge (i-1,j)-(i+1,j).
			coarse = 0.5 * (pool_pos(slot, interior_L(i - 1, j)) + pool_pos(slot, interior_L(i + 1, j)));
		} else if (ip == 0 && jp == 1) {
			// Midpoint of the vertical coarse edge (i,j-1)-(i,j+1).
			coarse = 0.5 * (pool_pos(slot, interior_L(i, j - 1)) + pool_pos(slot, interior_L(i, j + 1)));
		} else {
			// Both odd: midpoint of the diagonal coarse edge (i+1,j-1)-(i-1,j+1).
			coarse = 0.5 * (pool_pos(slot, interior_L(i + 1, j - 1)) + pool_pos(slot, interior_L(i - 1, j + 1)));
		}
	} else {
		// Skirt vertex along-edge position t: odd t morphs to the midpoint of the
		// two even skirt verts, so the skirt stays glued to its morphing edge.
		int s = VERTEX_ID - interior_count;
		int edge = s / (chunk_res + 1);
		int t = s - edge * (chunk_res + 1);
		int base = interior_count + edge * (chunk_res + 1);
		coarse = (t % 2 == 0)
			? fine
			: 0.5 * (pool_pos(slot, base + (t - 1)) + pool_pos(slot, base + (t + 1)));
	}

	VERTEX = mix(coarse, fine, morph);
	// Large-world jitter fix: the built-in pipeline computes view_pos as
	// R*VERTEX + t with VERTEX at planet magnitude (~6371 km); the f32 rounding
	// of R*VERTEX (~0.4 m) re-randomises every frame the view matrix changes —
	// visible ground wobble when ROTATING the camera up close (stable when
	// still, because the rounding is then constant). Overriding POSITION with
	// R*(VERTEX - camera) keeps the rotated quantity small near the camera, so
	// close-range precision is full while distant terrain is unaffected.
	vec3 rel = VERTEX - CAMERA_POSITION_WORLD;
	POSITION = PROJECTION_MATRIX * vec4(mat3(VIEW_MATRIX) * rel, 1.0);
	// Pass the slot + chunk-local bary (UV) to the fragment; colour + normal now
	// come per-pixel from the detail atlas baked by ChunkTileBake.slang.
	v_slot = slot;
	v_uv = UV;
}

// One atlas texel of THIS chunk's tile at integer (tx, ty), clamped into the
// baked lower-left triangle (ty <= tile_res-1-tx) so we never read an unbaked or
// cross-slot texel (the upper-right half is garbage; tiles are 1-D runs wrapped
// at attr_w, so neighbours abut in memory).
vec3 atlas_texel(sampler2D atlas, int tx, int ty) {
	tx = clamp(tx, 0, tile_res - 1);
	ty = clamp(ty, 0, tile_res - 1 - tx);
	int g = v_slot * tile_res * tile_res + ty * tile_res + tx;
	return texelFetch(atlas, ivec2(g % attr_w, g / attr_w), 0).rgb;
}

// Manual bilinear across the per-chunk tile. Hardware filter_linear can't be used
// (the tile isn't a contiguous 2-D rect and its hypotenuse abuts unbaked texels),
// so we tap the 4 surrounding texels and lerp, each tap clamped into the valid
// region — turning the low-res far tiles into smooth gradients instead of blocks.
// (Detail still sharpens as the quadtree subdivides nearer the camera.)
vec3 atlas_bilinear(sampler2D atlas) {
	vec2 tf = v_uv * float(tile_res) - 0.5; // texel space, sample at texel centres
	vec2 fr = clamp(fract(tf), 0.0, 1.0);
	int x0 = int(floor(tf.x));
	int y0 = int(floor(tf.y));
	vec3 c0 = mix(atlas_texel(atlas, x0, y0),     atlas_texel(atlas, x0 + 1, y0),     fr.x);
	vec3 c1 = mix(atlas_texel(atlas, x0, y0 + 1), atlas_texel(atlas, x0 + 1, y0 + 1), fr.x);
	return mix(c0, c1, fr.y);
}

void fragment() {
	ALBEDO = lod_colors ? hue_color(v_slot) : atlas_bilinear(color_atlas);

	// Decode the world-space detail normal (stored *0.5+0.5) and convert to view.
	// Bilinear-then-normalize is the standard normal-map filter.
	vec3 wn = normalize(atlas_bilinear(normal_atlas) * 2.0 - 1.0);
	NORMAL = normalize((VIEW_MATRIX * vec4(wn, 0.0)).xyz);
	ROUGHNESS = 1.0;
}
