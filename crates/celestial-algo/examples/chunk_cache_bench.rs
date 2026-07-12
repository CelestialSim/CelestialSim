//! Repro for the "cache full → low FPS" bug: time `ChunkCache::update` when the
//! cache is full and each frame brings new misses (forcing eviction).
//! Run: cargo run --release -p celestialsim-algo --example chunk_cache_bench

use std::time::Instant;

use celestial_algo::chunk_cache::ChunkCache;
use celestial_algo::quadtree::{Bary, Chunk, ChunkId};
use godot::builtin::Vector3;

fn chunk(path: u64) -> Chunk {
    let b = Bary { wb: 0.0, wc: 0.0 };
    Chunk {
        id: ChunkId { face: 0, depth: 10, path },
        bary: [b, b, b],
        corners: [Vector3::ZERO; 3],
        level: 10,
    }
}

fn bench(budget: u32, misses_per_frame: u64, frames: u64) {
    let mut cache = ChunkCache::new(budget);
    // Fill the cache to exactly `budget` distinct chunks.
    let fill: Vec<Chunk> = (0..budget as u64).map(chunk).collect();
    cache.update(&fill);

    // Each frame: a cut of all-new chunks (worst case — every chunk is a miss,
    // each forcing an eviction). `next` keeps advancing so ids never repeat.
    let mut next = budget as u64;
    let t = Instant::now();
    for _ in 0..frames {
        let cut: Vec<Chunk> = (0..misses_per_frame).map(|k| chunk(next + k)).collect();
        next += misses_per_frame;
        let _ = cache.update(&cut);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    println!(
        "budget={budget:>6}  misses/frame={misses_per_frame:>4}  => {ms:8.3} ms/update",
    );
}

fn main() {
    println!("== ChunkCache::update timing when FULL (eviction path) ==");
    for &budget in &[1024u32, 8192, 32768] {
        bench(budget, 500, 200);
    }
}
