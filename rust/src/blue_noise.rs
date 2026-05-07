//! 3D blue-noise mask generator.
//!
//! Adapted from `BlueNoise.py` by Christoph Peters
//! (<https://github.com/MomentsInGraphics/BlueNoise>) — © 2016 Christoph Peters,
//! redistributed under the CC0 1.0 Universal Public Domain Dedication bundled
//! in that repository. See `LICENSE-BLUE-NOISE` for the full license text.
//!
//! Implements the void-and-cluster method (Ulichney, 1993) on an N-dimensional
//! toroidal grid using FFT-based Gaussian convolution. Output is a packed
//! `Vec<u8>` of length `nx * ny * nz` where each byte is the dither rank,
//! linearly remapped from `0..nRank` into `0..=255`.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Default seed for the project's blue-noise volume. Bumping this invalidates
/// every cache file.
pub const DEFAULT_SEED: u64 = 0x00CE_1E54;

/// Standard deviation for the Gaussian energy filter, in pixels. The upstream
/// Python defaults the parameter to 1.5 but uses 1.9 in every example for the
/// 3D textures it actually ships, so we match the example value.
pub const DEFAULT_SIGMA: f32 = 1.9;

/// Fraction of cells initially set to "true" before iterative refinement. Must
/// be in `(0, 0.5)`. Matches the Python default.
pub const DEFAULT_INITIAL_SEED_FRACTION: f32 = 0.1;

/// Magic bytes guarding the on-disk cache format. Bump (e.g. to `BNS2`) if the
/// header layout or downstream byte semantics change.
const CACHE_MAGIC: &[u8; 4] = b"BNS1";
const CACHE_HEADER_LEN: usize = 32;

/// Generates a 3D blue-noise volume of the given size, seeded for determinism.
///
/// Output is `size[0] * size[1] * size[2]` bytes; each byte is a rank value
/// (`0..=255`) such that thresholding the volume at any value `t` produces a
/// blue-noise binary pattern with density approximately `t / 255`.
pub fn generate(size: [u32; 3], seed: u64) -> Vec<u8> {
    generate_with_params(size, seed, DEFAULT_SIGMA, DEFAULT_INITIAL_SEED_FRACTION)
}

/// Same as [`generate`] but exposes the algorithm tuning parameters.
pub fn generate_with_params(
    size: [u32; 3],
    seed: u64,
    sigma: f32,
    initial_seed_fraction: f32,
) -> Vec<u8> {
    let dims = [size[0] as usize, size[1] as usize, size[2] as usize];
    let n_rank = dims[0] * dims[1] * dims[2];
    assert!(n_rank > 0, "blue-noise volume must be non-empty");
    assert!(
        (0.0..0.5).contains(&initial_seed_fraction),
        "initial_seed_fraction must be in (0, 0.5)"
    );

    let dither = void_and_cluster(dims, seed, sigma, initial_seed_fraction);

    // Linear remap rank in [0, n_rank) to byte in [0, 256). Matches
    // `StoreNoiseTextureLDR`: `(rank * 256) // n_rank`.
    let mut bytes = vec![0u8; n_rank];
    for (i, rank) in dither.iter().enumerate() {
        let scaled = (*rank as u64 * 256) / n_rank as u64;
        bytes[i] = scaled.min(255) as u8;
    }
    bytes
}

/// Loads from `cache_path` if the file exists and matches the requested
/// `(size, seed)`; otherwise generates fresh bytes and writes the cache file
/// (creating parent directories if needed).
///
/// Cache file format:
/// * bytes 0..4   : magic `BNS1`
/// * bytes 4..16  : little-endian `(size_x, size_y, size_z)`
/// * bytes 16..24 : little-endian `seed_u64`
/// * bytes 24..32 : little-endian `payload_len_u64`
/// * bytes 32..   : raw R8 bytes, length `payload_len_u64`
pub fn load_or_generate(cache_path: &Path, size: [u32; 3], seed: u64) -> Vec<u8> {
    if let Some(bytes) = try_read_cache(cache_path, size, seed) {
        return bytes;
    }

    let bytes = generate(size, seed);
    let _ = write_cache(cache_path, size, seed, &bytes);
    bytes
}

fn try_read_cache(path: &Path, size: [u32; 3], seed: u64) -> Option<Vec<u8>> {
    let mut file = fs::File::open(path).ok()?;
    let mut header = [0u8; CACHE_HEADER_LEN];
    file.read_exact(&mut header).ok()?;
    if &header[0..4] != CACHE_MAGIC {
        return None;
    }

    let sx = read_u32(&header, 4)?;
    let sy = read_u32(&header, 8)?;
    let sz = read_u32(&header, 12)?;
    let cached_seed = read_u64(&header, 16)?;
    let cached_payload_len = read_u64(&header, 24)?;
    let expected_payload_len = expected_payload_len(size)?;

    if (sx, sy, sz) != (size[0], size[1], size[2]) {
        return None;
    }
    if cached_seed != seed {
        return None;
    }
    if cached_payload_len != expected_payload_len {
        return None;
    }

    let total_expected_len = CACHE_HEADER_LEN as u64 + cached_payload_len;
    let metadata = fs::metadata(path).ok()?;
    if metadata.len() != total_expected_len {
        return None;
    }

    let mut bytes = vec![0u8; cached_payload_len as usize];
    file.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

fn write_cache(path: &Path, size: [u32; 3], seed: u64, bytes: &[u8]) -> io::Result<()> {
    let expected_len = expected_payload_len(size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "cache size overflow"))?;
    if bytes.len() as u64 != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "payload length does not match requested cache dimensions",
        ));
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut header = [0u8; CACHE_HEADER_LEN];
    header[0..4].copy_from_slice(CACHE_MAGIC);
    header[4..8].copy_from_slice(&size[0].to_le_bytes());
    header[8..12].copy_from_slice(&size[1].to_le_bytes());
    header[12..16].copy_from_slice(&size[2].to_le_bytes());
    header[16..24].copy_from_slice(&seed.to_le_bytes());
    header[24..32].copy_from_slice(&expected_len.to_le_bytes());

    let mut file = fs::File::create(path)?;
    file.write_all(&header)?;
    file.write_all(bytes)?;
    Ok(())
}

fn expected_payload_len(size: [u32; 3]) -> Option<u64> {
    (size[0] as u64)
        .checked_mul(size[1] as u64)?
        .checked_mul(size[2] as u64)
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset + 4)?;
    let mut raw = [0u8; 4];
    raw.copy_from_slice(slice);
    Some(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let slice = bytes.get(offset..offset + 8)?;
    let mut raw = [0u8; 8];
    raw.copy_from_slice(slice);
    Some(u64::from_le_bytes(raw))
}

// ---------------------------------------------------------------------------
// Void-and-cluster algorithm
// ---------------------------------------------------------------------------

fn void_and_cluster(
    dims: [usize; 3],
    seed: u64,
    sigma: f32,
    initial_seed_fraction: f32,
) -> Vec<u32> {
    let n_rank = dims[0] * dims[1] * dims[2];
    let n_initial_one = (n_rank as f32 * initial_seed_fraction)
        .floor()
        .max(1.0)
        .min(((n_rank as i64 - 1) / 2) as f32) as usize;

    // --- Initial random pattern ---
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut perm: Vec<u32> = (0..n_rank as u32).collect();
    perm.shuffle(&mut rng);
    let mut pattern = vec![false; n_rank];
    for &idx in &perm[0..n_initial_one] {
        pattern[idx as usize] = true;
    }

    // --- Cache the Gaussian frequency-domain kernel (toroidal) ---
    let kernel = gaussian_freq_kernel(dims, sigma);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_rank);
    let ifft = planner.plan_fft_inverse(n_rank);
    // We re-use a single big 1D FFT over the flattened buffer with a per-axis
    // strategy below — but for simplicity (and because dims are small in
    // practice — 64³) we instead do a true N-D FFT manually.
    drop(fft);
    drop(ifft);

    // --- Iteratively swap tightest cluster -> largest void ---
    loop {
        let i_cluster = find_tightest_cluster(&pattern, dims, &kernel);
        pattern[i_cluster] = false;
        let i_void = find_largest_void(&pattern, dims, &kernel);
        if i_void == i_cluster {
            pattern[i_cluster] = true;
            break;
        }
        pattern[i_void] = true;
    }

    let initial_pattern = pattern.clone();
    let mut dither = vec![0u32; n_rank];

    // --- Phase 1: rank existing minority pixels ---
    let mut work = pattern.clone();
    for rank in (0..n_initial_one).rev() {
        let i = find_tightest_cluster(&work, dims, &kernel);
        work[i] = false;
        dither[i] = rank as u32;
    }

    // --- Phase 2: rank the remaining majority pixels by void filling ---
    work = initial_pattern;
    let half = (n_rank + 1) / 2;
    for rank in n_initial_one..half {
        let i = find_largest_void(&work, dims, &kernel);
        work[i] = true;
        dither[i] = rank as u32;
    }

    // --- Phase 3: rank the second half by the inverted-pattern logic ---
    for rank in half..n_rank {
        let i = find_tightest_cluster(&work, dims, &kernel);
        work[i] = true;
        dither[i] = rank as u32;
    }

    dither
}

/// Returns the flat index of the largest void in `pattern`. Matches
/// `FindLargestVoid` in the Python source: invert if majority is `true`, then
/// among true cells (in the working sense — the minority) find the cell with
/// the lowest filtered density.
fn find_largest_void(pattern: &[bool], dims: [usize; 3], kernel: &[f32]) -> usize {
    let (working, inverted) = canonicalize(pattern);
    let filtered = gaussian_filter(&working, dims, kernel);
    // Largest void = index with smallest filter response *among majority cells*
    // in the original pattern (i.e. the cells where `pattern[i]` matches the
    // majority value). In `working` those are the `false` cells.
    // The Python sets `BinaryPattern? 2.0 : filtered`. argmin then picks among
    // false cells.
    let _ = inverted;
    let mut best_idx = 0usize;
    let mut best_val = f32::INFINITY;
    for (i, &w) in working.iter().enumerate() {
        let v = if w { 2.0 } else { filtered[i] };
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Returns the flat index of the tightest cluster — matches `FindTightestCluster`.
fn find_tightest_cluster(pattern: &[bool], dims: [usize; 3], kernel: &[f32]) -> usize {
    let (working, _) = canonicalize(pattern);
    let filtered = gaussian_filter(&working, dims, kernel);
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &w) in working.iter().enumerate() {
        let v = if w { filtered[i] } else { -1.0 };
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Returns the canonicalised pattern (with `true` always the minority class)
/// alongside a flag indicating whether it was inverted.
fn canonicalize(pattern: &[bool]) -> (Vec<bool>, bool) {
    let ones = pattern.iter().filter(|b| **b).count();
    if ones * 2 >= pattern.len() {
        (pattern.iter().map(|b| !*b).collect(), true)
    } else {
        (pattern.to_vec(), false)
    }
}

/// Builds a 3-D Gaussian kernel pre-evaluated in frequency space. Element
/// `[kx, ky, kz]` equals `exp(-0.5 * (2*pi*sigma)^2 * (fx^2 + fy^2 + fz^2))`
/// where `fx = (kx <= N/2 ? kx : kx - N) / N`. This matches the convention of
/// `scipy.ndimage.fourier_gaussian` (sigma in pixels of the spatial filter).
fn gaussian_freq_kernel(dims: [usize; 3], sigma: f32) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut k = vec![0.0f32; nx * ny * nz];
    let coeff = -0.5 * (2.0 * std::f32::consts::PI * sigma).powi(2);
    let freq = |k: usize, n: usize| -> f32 {
        let kk = if k * 2 <= n {
            k as i64
        } else {
            k as i64 - n as i64
        };
        kk as f32 / n as f32
    };
    for kxi in 0..nx {
        let fx = freq(kxi, nx);
        for kyi in 0..ny {
            let fy = freq(kyi, ny);
            for kzi in 0..nz {
                let fz = freq(kzi, nz);
                let r2 = fx * fx + fy * fy + fz * fz;
                k[kxi * ny * nz + kyi * nz + kzi] = (coeff * r2).exp();
            }
        }
    }
    k
}

/// 3-D toroidal Gaussian convolution via FFT.
///
/// Performs `IFFT( FFT(input) * kernel )` taking the real part, where `kernel`
/// is the pre-computed frequency-domain Gaussian from
/// [`gaussian_freq_kernel`].
fn gaussian_filter(input: &[bool], dims: [usize; 3], kernel: &[f32]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    let mut buf: Vec<Complex<f32>> = input
        .iter()
        .map(|b| Complex::new(if *b { 1.0 } else { 0.0 }, 0.0))
        .collect();

    fft_3d_forward(&mut buf, dims);

    for i in 0..n {
        buf[i] *= kernel[i];
    }

    fft_3d_inverse(&mut buf, dims);

    let inv_n = 1.0 / n as f32;
    buf.into_iter().map(|c| c.re * inv_n).collect()
}

/// In-place 3D FFT over a buffer laid out in `[x][y][z]` row-major order.
fn fft_3d_forward(buf: &mut [Complex<f32>], dims: [usize; 3]) {
    fft_along_axis(buf, dims, 0, false);
    fft_along_axis(buf, dims, 1, false);
    fft_along_axis(buf, dims, 2, false);
}

fn fft_3d_inverse(buf: &mut [Complex<f32>], dims: [usize; 3]) {
    fft_along_axis(buf, dims, 0, true);
    fft_along_axis(buf, dims, 1, true);
    fft_along_axis(buf, dims, 2, true);
}

/// Runs a 1D FFT along one axis of a 3D row-major buffer.
fn fft_along_axis(buf: &mut [Complex<f32>], dims: [usize; 3], axis: usize, inverse: bool) {
    let [nx, ny, nz] = dims;
    let n = match axis {
        0 => nx,
        1 => ny,
        2 => nz,
        _ => unreachable!(),
    };
    let mut planner = FftPlanner::<f32>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let mut line = vec![Complex::new(0.0, 0.0); n];

    let stride = match axis {
        0 => ny * nz,
        1 => nz,
        2 => 1,
        _ => unreachable!(),
    };

    // Iterate over all (other_axis_a, other_axis_b) pairs.
    let (outer_a, outer_b) = match axis {
        0 => (ny, nz),
        1 => (nx, nz),
        2 => (nx, ny),
        _ => unreachable!(),
    };
    for a in 0..outer_a {
        for b in 0..outer_b {
            let base = match axis {
                0 => a * nz + b,
                1 => a * (ny * nz) + b,
                2 => a * (ny * nz) + b * nz,
                _ => unreachable!(),
            };
            for i in 0..n {
                line[i] = buf[base + i * stride];
            }
            fft.process_with_scratch(&mut line, &mut scratch);
            for i in 0..n {
                buf[base + i * stride] = line[i];
            }
        }
    }
}
