//! Binary dump format for the LOD pipeline's final visible mesh, used by the
//! no-holes integration test. The producer is a #[func] on CesCelestialRust;
//! the consumer is `rust/tests/test_no_holes.rs`. Format intentionally minimal —
//! it carries only what the geometric T-junction test needs.
//!
//! Layout (little-endian):
//!   [4]  magic = b"CSDP"
//!   [4]  version: u32 (currently 1)
//!   [4]  n_visible_tris: u32
//!   [4]  radius: f32
//!   per-tri (n_visible_tris records, 52 bytes each):
//!     [36]  positions: 3 × [f32;3]  (post-stitch v0, v1, v2)
//!     [4]   level: i32
//!     [12]  neighbors: 3 × i32      (visible-index of neighbor across edges
//!                                    ab, bc, ca; -1 if neighbor is not visible
//!                                    or out of range)

pub const MAGIC: [u8; 4] = *b"CSDP";
pub const VERSION: u32 = 1;
pub const TRI_RECORD_BYTES: usize = 36 + 4 + 12;
pub const HEADER_BYTES: usize = 4 + 4 + 4 + 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DumpHeader {
    pub n_visible_tris: u32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DumpedTri {
    pub positions: [[f32; 3]; 3],
    pub level: i32,
    pub neighbors: [i32; 3],
}

#[derive(Debug, PartialEq)]
pub enum DumpError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated { expected: usize, actual: usize },
}

pub fn encode(header: DumpHeader, tris: &[DumpedTri]) -> Vec<u8> {
    assert_eq!(
        header.n_visible_tris as usize,
        tris.len(),
        "header.n_visible_tris must match tris.len()"
    );
    let mut out = Vec::with_capacity(HEADER_BYTES + tris.len() * TRI_RECORD_BYTES);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&header.n_visible_tris.to_le_bytes());
    out.extend_from_slice(&header.radius.to_le_bytes());
    for tri in tris {
        for v in &tri.positions {
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.extend_from_slice(&tri.level.to_le_bytes());
        for n in &tri.neighbors {
            out.extend_from_slice(&n.to_le_bytes());
        }
    }
    out
}

pub fn decode(bytes: &[u8]) -> Result<(DumpHeader, Vec<DumpedTri>), DumpError> {
    if bytes.len() < HEADER_BYTES {
        return Err(DumpError::Truncated {
            expected: HEADER_BYTES,
            actual: bytes.len(),
        });
    }
    if bytes[0..4] != MAGIC {
        return Err(DumpError::BadMagic);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(DumpError::UnsupportedVersion(version));
    }
    let n_visible_tris = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let radius = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let expected = HEADER_BYTES + (n_visible_tris as usize) * TRI_RECORD_BYTES;
    if bytes.len() < expected {
        return Err(DumpError::Truncated {
            expected,
            actual: bytes.len(),
        });
    }
    let mut tris = Vec::with_capacity(n_visible_tris as usize);
    let mut o = HEADER_BYTES;
    for _ in 0..n_visible_tris {
        let mut positions = [[0.0_f32; 3]; 3];
        for v in 0..3 {
            for c in 0..3 {
                positions[v][c] = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
                o += 4;
            }
        }
        let level = i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        o += 4;
        let mut neighbors = [0_i32; 3];
        for n in &mut neighbors {
            *n = i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            o += 4;
        }
        tris.push(DumpedTri {
            positions,
            level,
            neighbors,
        });
    }
    Ok((
        DumpHeader {
            n_visible_tris,
            radius,
        },
        tris,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let header = DumpHeader {
            n_visible_tris: 0,
            radius: 300.0,
        };
        let bytes = encode(header, &[]);
        assert_eq!(bytes.len(), HEADER_BYTES);
        let (h, t) = decode(&bytes).unwrap();
        assert_eq!(h, header);
        assert!(t.is_empty());
    }

    #[test]
    fn round_trip_two_tris() {
        let tris = vec![
            DumpedTri {
                positions: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                level: 2,
                neighbors: [1, -1, 0],
            },
            DumpedTri {
                positions: [
                    [-1.5, 0.0, 0.5],
                    [0.0, -1.5, 0.5],
                    [0.0, 0.0, -1.5],
                ],
                level: 5,
                neighbors: [-1, 0, -1],
            },
        ];
        let header = DumpHeader {
            n_visible_tris: tris.len() as u32,
            radius: 42.5,
        };
        let bytes = encode(header, &tris);
        assert_eq!(bytes.len(), HEADER_BYTES + 2 * TRI_RECORD_BYTES);
        let (h, t) = decode(&bytes).unwrap();
        assert_eq!(h, header);
        assert_eq!(t, tris);
    }

    #[test]
    fn decode_bad_magic_rejected() {
        let mut bytes = encode(
            DumpHeader {
                n_visible_tris: 0,
                radius: 1.0,
            },
            &[],
        );
        bytes[0] = b'X';
        assert_eq!(decode(&bytes), Err(DumpError::BadMagic));
    }

    #[test]
    fn decode_truncated_header_rejected() {
        let bytes = vec![b'C', b'S', b'D']; // 3 bytes, less than HEADER_BYTES
        assert!(matches!(
            decode(&bytes),
            Err(DumpError::Truncated { .. })
        ));
    }

    #[test]
    fn decode_truncated_body_rejected() {
        let header = DumpHeader {
            n_visible_tris: 2,
            radius: 1.0,
        };
        let tris = vec![DumpedTri {
            positions: [[0.0; 3]; 3],
            level: 0,
            neighbors: [0; 3],
        }];
        // Manually encode header claiming 2 tris but body only has 1.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&header.n_visible_tris.to_le_bytes());
        bytes.extend_from_slice(&header.radius.to_le_bytes());
        for t in &tris {
            for v in &t.positions {
                for c in v {
                    bytes.extend_from_slice(&c.to_le_bytes());
                }
            }
            bytes.extend_from_slice(&t.level.to_le_bytes());
            for n in &t.neighbors {
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        }
        assert!(matches!(
            decode(&bytes),
            Err(DumpError::Truncated { .. })
        ));
    }

    #[test]
    fn decode_unsupported_version_rejected() {
        let mut bytes = encode(
            DumpHeader {
                n_visible_tris: 0,
                radius: 1.0,
            },
            &[],
        );
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        assert_eq!(decode(&bytes), Err(DumpError::UnsupportedVersion(999)));
    }
}
