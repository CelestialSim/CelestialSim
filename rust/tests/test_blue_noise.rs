use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

#[path = "../src/blue_noise.rs"]
mod blue_noise;

use blue_noise::{generate, load_or_generate};

#[test]
fn blue_noise_generate_deterministic_for_seed() {
    let a = generate([8, 8, 8], 42);
    let b = generate([8, 8, 8], 42);
    assert_eq!(a, b);
}

#[test]
fn blue_noise_generate_correct_size() {
    let size = [7, 9, 11];
    let bytes = generate(size, 1234);
    assert_eq!(bytes.len(), (size[0] * size[1] * size[2]) as usize);
}

#[test]
fn blue_noise_value_distribution_uniform() {
    let bytes = generate([8, 8, 8], 0xCE1E54);
    let mut counts = [0u32; 256];
    for &b in &bytes {
        counts[b as usize] += 1;
    }

    let min_count = *counts.iter().min().unwrap();
    let max_count = *counts.iter().max().unwrap();
    assert!(
        min_count >= 1,
        "every byte value should appear at least once, min={}",
        min_count
    );
    assert!(
        max_count <= 4,
        "no byte value should appear more than 4 times, max={}",
        max_count
    );
}

#[test]
#[ignore]
fn blue_noise_high_frequency_energy_dominates() {
    let dims = [16usize, 16usize, 16usize];
    let bytes = generate([16, 16, 16], 1337);

    let mean = bytes.iter().map(|&b| b as f32).sum::<f32>() / bytes.len() as f32;
    let mut spectrum: Vec<Complex<f32>> = bytes
        .iter()
        .map(|&b| Complex::new((b as f32 - mean) / 255.0, 0.0))
        .collect();

    fft_3d_forward(&mut spectrum, dims);

    let [nx, ny, nz] = dims;
    let mut low_energy = 0.0f32;
    let mut high_energy = 0.0f32;

    for x in 0..nx {
        let fx = wrapped_freq(x, nx);
        for y in 0..ny {
            let fy = wrapped_freq(y, ny);
            for z in 0..nz {
                let fz = wrapped_freq(z, nz);
                let r = (fx * fx + fy * fy + fz * fz).sqrt();
                if r == 0.0 {
                    continue;
                }
                let power = spectrum[x * ny * nz + y * nz + z].norm_sqr();
                if r <= 0.12 {
                    low_energy += power;
                }
                if (0.28..=0.5).contains(&r) {
                    high_energy += power;
                }
            }
        }
    }

    assert!(high_energy > 0.0, "high-band energy must be non-zero");
    assert!(
        low_energy < high_energy * 0.3,
        "expected low-band energy ({}) to be < 30% of high-band energy ({})",
        low_energy,
        high_energy
    );
}

#[test]
fn load_or_generate_writes_then_reads() {
    let dir = unique_temp_dir("ces_blue_noise_it");
    let path = dir.join("noise.bin");

    let first = load_or_generate(&path, [8, 8, 8], 7);
    assert!(path.exists(), "cache file should be created on first call");

    // Mutate one payload byte in-place to verify second call really reads cache.
    let mut raw = fs::read(&path).unwrap();
    let last = raw.len() - 1;
    raw[last] ^= 0xFF;
    fs::write(&path, &raw).unwrap();

    let mut expected = first.clone();
    let expected_last = expected.len() - 1;
    expected[expected_last] ^= 0xFF;

    let second = load_or_generate(&path, [8, 8, 8], 7);
    assert_eq!(second, expected);

    let _ = fs::remove_dir_all(&dir);
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", prefix, nanos))
}

fn wrapped_freq(k: usize, n: usize) -> f32 {
    let signed = if k <= n / 2 {
        k as isize
    } else {
        k as isize - n as isize
    };
    signed as f32 / n as f32
}

fn fft_3d_forward(buf: &mut [Complex<f32>], dims: [usize; 3]) {
    fft_along_axis(buf, dims, 0);
    fft_along_axis(buf, dims, 1);
    fft_along_axis(buf, dims, 2);
}

fn fft_along_axis(buf: &mut [Complex<f32>], dims: [usize; 3], axis: usize) {
    let [nx, ny, nz] = dims;
    let n = match axis {
        0 => nx,
        1 => ny,
        2 => nz,
        _ => unreachable!(),
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let mut line = vec![Complex::new(0.0, 0.0); n];

    let stride = match axis {
        0 => ny * nz,
        1 => nz,
        2 => 1,
        _ => unreachable!(),
    };

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
