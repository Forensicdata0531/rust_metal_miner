// src/mitm.rs
//! MITM / Pollard-Rho state helpers and Metal buffer utilities.
//!
//! The GPU kernel receives `device uint* mitm_states` (flat u32 array).
//! This module provides a Rust-side `RhoState` and a fixed packing of each
//! RhoState into u32 words suitable for placing in a Metal buffer.
//!
//! Layout per RhoState (u32 words):
//!  0: seed_low  (u32)
//!  1: seed_high (u32)      -> seed = seed_low | (seed_high << 32)
//!  2..9: value (8 * u32)   -> 32 bytes (value[0..4] -> u32[2], etc. little-endian)
//! 10: steps_low  (u32)
//! 11: steps_high (u32)     -> steps = steps_low | (steps_high << 32)
use std::convert::TryInto;
use std::sync::Arc;

use metal::{Buffer, Device, MTLResourceOptions};

/// Number of u32 words used to represent a single RhoState in the GPU buffer.
/// Update this if your kernel's layout changes.
pub const MITM_STATE_U32_WORDS: usize = 12;

/// Number of bytes per RhoState (u32 words * 4 bytes)
pub const MITM_STATE_BYTES: usize = MITM_STATE_U32_WORDS * 4;

/// RhoState stored on the CPU side
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RhoState {
    /// 64-bit seed / identifier for the rho chain
    pub seed: u64,
    /// 32-byte value (digest or point)
    pub value: [u8; 32],
    /// number of steps / iterations taken so far
    pub steps: u64,
}

impl RhoState {
    /// Create an "empty" RhoState
    pub fn new(seed: u64, value: [u8; 32], steps: u64) -> Self {
        Self { seed, value, steps }
    }

    /// Zero state
    pub fn zero() -> Self {
        Self { seed: 0, value: [0u8; 32], steps: 0 }
    }
}

/// Serialize a slice of `RhoState` into a Vec<u32> using the fixed layout.
/// The returned Vec length = states.len() * MITM_STATE_U32_WORDS.
pub fn serialize_rho_states_to_u32(states: &[RhoState]) -> Vec<u32> {
    let mut out = Vec::with_capacity(states.len() * MITM_STATE_U32_WORDS);
    for s in states {
        // seed (u64) -> two u32s (little-endian)
        out.push(s.seed as u32);
        out.push((s.seed >> 32) as u32);

        // value: 32 bytes -> 8 u32 words, each u32 is little-endian from 4 bytes
        for chunk in s.value.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            out.push(u32::from_le_bytes(arr));
        }

        // steps (u64) -> two u32s
        out.push(s.steps as u32);
        out.push((s.steps >> 32) as u32);
    }
    out
}

/// Deserialize a u32 slice (GPU buffer contents) into a Vec<RhoState>.
/// If the provided slice length is not a multiple of the state words, the tail is ignored.
pub fn deserialize_u32_to_rho_states(slice: &[u32]) -> Vec<RhoState> {
    let mut out = Vec::with_capacity(slice.len() / MITM_STATE_U32_WORDS);
    let mut idx = 0usize;
    while idx + MITM_STATE_U32_WORDS <= slice.len() {
        let seed_low = slice[idx];
        let seed_high = slice[idx + 1];
        let seed = (seed_low as u64) | ((seed_high as u64) << 32);

        let mut value = [0u8; 32];
        for w in 0..8 {
            let word = slice[idx + 2 + w];
            let be_bytes = word.to_le_bytes(); // words are stored little-endian
            value[w * 4..(w + 1) * 4].copy_from_slice(&be_bytes);
        }

        let steps_low = slice[idx + 10];
        let steps_high = slice[idx + 11];
        let steps = (steps_low as u64) | ((steps_high as u64) << 32);

        out.push(RhoState { seed, value, steps });
        idx += MITM_STATE_U32_WORDS;
    }
    out
}

/// Allocate a Metal buffer sized to hold `count` RhoStates with `StorageModeShared`.
/// Returns an `Arc<Buffer>` for convenient sharing.
pub fn create_mitm_buffer(device: &Device, count: usize) -> Arc<Buffer> {
    let bytes = (count * MITM_STATE_BYTES) as u64;
    Arc::new(device.new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}

/// Write `states` into the provided Metal `buffer` (which must be large enough).
/// This performs a direct memory write into the buffer `contents()` area.
pub unsafe fn write_rho_states_to_buffer(buffer: &Buffer, states: &[RhoState]) {
    let u32_slice = serialize_rho_states_to_u32(states);
    let dst = buffer.contents() as *mut u32;
    // ensure we don't overflow buffer: make it safe-ish by calculating available u32s
    let available_u32 = (buffer.length() as usize) / 4;
    let to_copy = u32_slice.len().min(available_u32);
    std::ptr::copy_nonoverlapping(u32_slice.as_ptr(), dst, to_copy);
    // zero remaining if buffer larger than provided states (optional — helpful)
    if to_copy < available_u32 {
        let zero_ptr = dst.add(to_copy);
        let remaining = available_u32 - to_copy;
        std::ptr::write_bytes(zero_ptr, 0, remaining);
    }
}

/// Read RhoStates from the provided Metal `buffer` and return Vec<RhoState>.
/// This will read up to `count` states (or fewer if buffer is smaller).
pub unsafe fn read_rho_states_from_buffer(buffer: &Buffer, count: usize) -> Vec<RhoState> {
    let available_u32 = (buffer.length() as usize) / 4;
    let want_u32 = count.saturating_mul(MITM_STATE_U32_WORDS);
    let read_words = available_u32.min(want_u32);
    let src = buffer.contents() as *const u32;
    let slice = std::slice::from_raw_parts(src, read_words);
    deserialize_u32_to_rho_states(slice)
}

/// Convenience: initialize a Metal mitm buffer with `count` zeroed states.
pub unsafe fn init_zeroed_mitm_buffer(device: &Device, count: usize) -> Arc<Buffer> {
    let buf = create_mitm_buffer(device, count);
    // zero the memory
    std::ptr::write_bytes(buf.contents(), 0, buf.length() as usize);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_and_deserialize_roundtrip() {
        let mut states = Vec::new();
        for i in 0..3u64 {
            let mut val = [0u8; 32];
            for j in 0..32 {
                val[j] = (i as u8).wrapping_add(j as u8);
            }
            states.push(RhoState::new(i * 0xFF00_FFu64, val, i * 12345));
        }

        let u32s = serialize_rho_states_to_u32(&states);
        let recovered = deserialize_u32_to_rho_states(&u32s);
        assert_eq!(states, recovered);
    }

    // This test is only run on host and doesn't touch Metal
    #[test]
    fn layout_word_count() {
        assert_eq!(MITM_STATE_U32_WORDS * 4, MITM_STATE_BYTES);
    }
}
