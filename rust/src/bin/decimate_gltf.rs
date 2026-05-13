// Standalone CLI: decimate every mesh primitive in a glTF using
// meshoptimizer's sloppy simplifier, producing a new .gltf + .bin next to
// the input file. Textures are left untouched (referenced via the same
// `textures/` subfolder).
//
// Usage:
//   cargo run --bin decimate_gltf -- <input.gltf> <ratio> [suffix=_lo]
//
//   ratio   target fraction of original triangles per primitive (e.g. 0.15)
//   suffix  appended to the input stem for the output filename (default _lo)
//
// Example:
//   cargo run --bin decimate_gltf -- \
//     addons/celestial_sim/assets/environment/rocks/rock_moss_set_02_1k/rock_moss_set_02_1k.gltf \
//     0.15
//
// Writes `rock_moss_set_02_1k_lo.gltf` and `rock_moss_set_02_1k_lo.bin` in
// the same directory; texture URIs in the new gltf still resolve to the
// existing `textures/` folder.

use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("Usage: decimate_gltf <input.gltf> <ratio> [suffix=_lo]");
        eprintln!("  ratio   target triangle fraction (e.g. 0.15 = keep 15%)");
        eprintln!("  suffix  appended to input stem for output (default _lo)");
        return ExitCode::from(2);
    }
    let input = PathBuf::from(&args[1]);
    let ratio: f32 = match args[2].parse() {
        Ok(r) if r > 0.0 && r <= 1.0 => r,
        _ => {
            eprintln!("ratio must be a float in (0, 1]");
            return ExitCode::from(2);
        }
    };
    let suffix = args.get(3).map(String::as_str).unwrap_or("_lo");

    match run(&input, ratio, suffix) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(input: &PathBuf, ratio: f32, suffix: &str) -> Result<(), Box<dyn std::error::Error>> {
    let json_text = fs::read_to_string(input)?;
    let mut gltf: Value = serde_json::from_str(&json_text)?;

    let buffer0 = gltf
        .get("buffers")
        .and_then(|b| b.get(0))
        .ok_or("buffers[0] missing")?;
    let bin_uri = buffer0
        .get("uri")
        .and_then(Value::as_str)
        .ok_or("buffers[0].uri missing (embedded buffers not supported)")?
        .to_string();

    let input_dir = input
        .parent()
        .ok_or("input path has no parent directory")?;
    let bin_path = input_dir.join(&bin_uri);
    let mut bin: Vec<u8> = fs::read(&bin_path)?;
    let original_bin_len = bin.len();

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("input has no usable file stem")?;
    let output_gltf = input_dir.join(format!("{stem}{suffix}.gltf"));
    let output_bin_name = format!("{stem}{suffix}.bin");
    let output_bin = input_dir.join(&output_bin_name);

    let mesh_count = gltf
        .get("meshes")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    let mut total_before: u32 = 0;
    let mut total_after: u32 = 0;

    for mesh_idx in 0..mesh_count {
        let prim_count = gltf["meshes"][mesh_idx]["primitives"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        for prim_idx in 0..prim_count {
            let prim = &gltf["meshes"][mesh_idx]["primitives"][prim_idx];
            let pos_acc_idx = prim["attributes"]["POSITION"]
                .as_u64()
                .ok_or("primitive without POSITION accessor")? as usize;
            let idx_acc_idx = prim["indices"]
                .as_u64()
                .ok_or("primitive without indices (non-indexed not supported)")?
                as usize;
            // Optional attributes — UVs and normals if present. We weight them
            // into the simplification cost so seams and shading aren't broken.
            let uv_acc_idx = prim["attributes"]["TEXCOORD_0"].as_u64().map(|n| n as usize);
            let nor_acc_idx = prim["attributes"]["NORMAL"].as_u64().map(|n| n as usize);

            let positions: Vec<[f32; 3]> = read_vec3_accessor(&gltf, pos_acc_idx, &bin)?;
            let indices = read_indices_accessor(&gltf, idx_acc_idx, &bin)?;
            let uvs: Option<Vec<[f32; 2]>> = uv_acc_idx
                .map(|i| read_vec2_accessor(&gltf, i, &bin))
                .transpose()?;
            let normals: Option<Vec<[f32; 3]>> = nor_acc_idx
                .map(|i| read_vec3_accessor(&gltf, i, &bin))
                .transpose()?;

            let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
            let adapter = meshopt::VertexDataAdapter::new(pos_bytes, 12, 0)
                .map_err(|e| format!("VertexDataAdapter::new failed: {e}"))?;
            let original_tris = (indices.len() / 3) as u32;
            let target_index_count = ((indices.len() as f32) * ratio).max(3.0) as usize;

            // Pack attributes into a flat [f32] interleaved per vertex. Order:
            //   [u, v, nx, ny, nz, …]
            // Weights mirror the order. UVs are weighted high (1.0) so the
            // simplifier doesn't fold across UV seams — that's what produced
            // the "river around the rock" stretching with simplify_sloppy.
            // Normals get a lower weight (0.5) so shading is preserved but
            // doesn't dominate over geometric simplification.
            let attr_stride = uvs.as_ref().map_or(0, |_| 2) + normals.as_ref().map_or(0, |_| 3);
            let new_indices: Vec<u32> = if attr_stride == 0 {
                meshopt::simplify(
                    &indices,
                    &adapter,
                    target_index_count,
                    f32::MAX,
                    meshopt::SimplifyOptions::None,
                    None,
                )
            } else {
                let n_verts = positions.len();
                let mut attrs: Vec<f32> = Vec::with_capacity(n_verts * attr_stride);
                let mut weights: Vec<f32> = Vec::with_capacity(attr_stride);
                if let Some(ref u) = uvs {
                    weights.extend_from_slice(&[1.0, 1.0]);
                    for v in 0..n_verts {
                        attrs.extend_from_slice(&u[v]);
                        if let Some(ref n) = normals {
                            attrs.extend_from_slice(&n[v]);
                        }
                    }
                    if normals.is_some() {
                        weights.extend_from_slice(&[0.5, 0.5, 0.5]);
                    }
                } else if let Some(ref n) = normals {
                    weights.extend_from_slice(&[0.5, 0.5, 0.5]);
                    for v in 0..n_verts {
                        attrs.extend_from_slice(&n[v]);
                    }
                }
                let locks = vec![false; positions.len()];
                meshopt::simplify_with_attributes_and_locks(
                    &indices,
                    &adapter,
                    &attrs,
                    &weights,
                    attr_stride * std::mem::size_of::<f32>(),
                    &locks,
                    target_index_count,
                    f32::MAX,
                    meshopt::SimplifyOptions::None,
                    None,
                )
            };
            let new_tris = (new_indices.len() / 3) as u32;
            total_before += original_tris;
            total_after += new_tris;

            // Append new indices to the bin as u32. Pad bin to 4-byte
            // alignment first (glTF requires bufferView byteOffset alignment
            // to match the accessor's component size).
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            let new_offset = bin.len();
            for &i in &new_indices {
                bin.extend_from_slice(&i.to_le_bytes());
            }
            let new_byte_length = new_indices.len() * 4;

            let new_bv = serde_json::json!({
                "buffer": 0,
                "byteOffset": new_offset,
                "byteLength": new_byte_length,
                "target": 34963_u32, // ELEMENT_ARRAY_BUFFER
            });
            let bvs = gltf["bufferViews"]
                .as_array_mut()
                .ok_or("bufferViews not an array")?;
            let new_bv_idx = bvs.len();
            bvs.push(new_bv);

            let new_acc = serde_json::json!({
                "bufferView": new_bv_idx,
                "componentType": 5125_u32, // UNSIGNED_INT
                "count": new_indices.len(),
                "type": "SCALAR",
            });
            let accs = gltf["accessors"]
                .as_array_mut()
                .ok_or("accessors not an array")?;
            let new_acc_idx = accs.len();
            accs.push(new_acc);

            gltf["meshes"][mesh_idx]["primitives"][prim_idx]["indices"] =
                Value::from(new_acc_idx);
        }
    }

    gltf["buffers"][0]["uri"] = Value::from(output_bin_name.clone());
    gltf["buffers"][0]["byteLength"] = Value::from(bin.len());

    fs::write(&output_bin, &bin)?;
    fs::write(&output_gltf, serde_json::to_string_pretty(&gltf)?)?;

    let pct_kept = if total_before > 0 {
        100.0 * total_after as f32 / total_before as f32
    } else {
        0.0
    };
    println!(
        "✓ wrote {}",
        output_gltf
            .strip_prefix(input_dir)
            .unwrap_or(&output_gltf)
            .display()
    );
    println!(
        "  triangles: {total_before} → {total_after} ({pct_kept:.1}% kept)",
    );
    println!(
        "  .bin grew {} → {} bytes (orphaned LOD0 indices retained)",
        original_bin_len,
        bin.len()
    );
    println!(
        "  textures: unchanged; the new .gltf points at the same `textures/` folder"
    );

    Ok(())
}

fn read_vec3_accessor(
    gltf: &Value,
    acc_idx: usize,
    bin: &[u8],
) -> Result<Vec<[f32; 3]>, Box<dyn std::error::Error>> {
    let acc = &gltf["accessors"][acc_idx];
    let count = acc["count"].as_u64().ok_or("accessor.count missing")? as usize;
    let comp = acc["componentType"].as_u64().unwrap_or(5126);
    if comp != 5126 {
        return Err(format!("POSITION accessor expects FLOAT (5126), got {comp}").into());
    }
    let bv_idx = acc["bufferView"]
        .as_u64()
        .ok_or("accessor without bufferView")? as usize;
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let bv = &gltf["bufferViews"][bv_idx];
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = bv["byteStride"].as_u64().unwrap_or(12) as usize;
    let start = bv_offset + acc_offset;

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = start + i * stride;
        let x = f32::from_le_bytes(bin[off..off + 4].try_into()?);
        let y = f32::from_le_bytes(bin[off + 4..off + 8].try_into()?);
        let z = f32::from_le_bytes(bin[off + 8..off + 12].try_into()?);
        out.push([x, y, z]);
    }
    Ok(out)
}

fn read_vec2_accessor(
    gltf: &Value,
    acc_idx: usize,
    bin: &[u8],
) -> Result<Vec<[f32; 2]>, Box<dyn std::error::Error>> {
    let acc = &gltf["accessors"][acc_idx];
    let count = acc["count"].as_u64().ok_or("accessor.count missing")? as usize;
    let comp = acc["componentType"].as_u64().unwrap_or(5126);
    if comp != 5126 {
        return Err(format!("vec2 accessor expects FLOAT (5126), got {comp}").into());
    }
    let bv_idx = acc["bufferView"]
        .as_u64()
        .ok_or("accessor without bufferView")? as usize;
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let bv = &gltf["bufferViews"][bv_idx];
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = bv["byteStride"].as_u64().unwrap_or(8) as usize;
    let start = bv_offset + acc_offset;

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = start + i * stride;
        let u = f32::from_le_bytes(bin[off..off + 4].try_into()?);
        let v = f32::from_le_bytes(bin[off + 4..off + 8].try_into()?);
        out.push([u, v]);
    }
    Ok(out)
}

fn read_indices_accessor(
    gltf: &Value,
    acc_idx: usize,
    bin: &[u8],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let acc = &gltf["accessors"][acc_idx];
    let count = acc["count"].as_u64().ok_or("accessor.count missing")? as usize;
    let comp = acc["componentType"].as_u64().ok_or("componentType missing")?;
    let bv_idx = acc["bufferView"]
        .as_u64()
        .ok_or("accessor without bufferView")? as usize;
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let bv = &gltf["bufferViews"][bv_idx];
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let start = bv_offset + acc_offset;

    match comp {
        5121 => Ok((0..count).map(|i| bin[start + i] as u32).collect()),
        5123 => Ok((0..count)
            .map(|i| u16::from_le_bytes([bin[start + i * 2], bin[start + i * 2 + 1]]) as u32)
            .collect()),
        5125 => {
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let off = start + i * 4;
                out.push(u32::from_le_bytes(bin[off..off + 4].try_into()?));
            }
            Ok(out)
        }
        _ => Err(format!("indices componentType {comp} not supported").into()),
    }
}
