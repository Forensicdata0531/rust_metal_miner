// ----------------- Constants -----------------
pub const INFLIGHT: usize = 8;
pub const BATCH_SIZE: usize = 65_536;

// number of worker threads per lane on the GPU (must divide NONCES_PER_THREAD)
pub const THREADS_PER_LANE: usize = 32;

// legacy name kept for compatibility
pub const THREADS_PER_GROUP: usize = THREADS_PER_LANE;

pub const LANES: usize = 4;
pub const DP_BITS: usize = 20;
pub const MAX_STEPS: u64 = 1_000_000;
pub const NUM_BUFFERS: usize = 8;
pub const NONCES_PER_THREAD: usize = 512;
pub const LANES_STATE_WORDS: usize = 8;
pub const NIBBLES: usize = 16;
