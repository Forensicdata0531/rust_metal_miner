// src/gpu.rs
use metal::*;
use tokio::sync::{mpsc, oneshot};

// Commands that can be sent to the GPU thread
pub enum GpuCommand {
    // Dispatch a compute kernel
    DispatchKernel {
        pipeline_name: String,
        buffer_count: usize,
        reply: oneshot::Sender<Result<(), String>>,
    },
}

// Spawns a dedicated OS thread that owns the Metal device and command queue.
// Returns a channel (`mpsc::UnboundedSender<GpuCommand>`) that the async runtime
// can use to schedule GPU work safely without crossing `Send` boundaries.
pub fn spawn_gpu_thread() -> mpsc::UnboundedSender<GpuCommand> {
    let (tx, mut rx) = mpsc::unbounded_channel::<GpuCommand>();

    std::thread::spawn(move || {
        // 🧠 This thread is now the only owner of Metal objects
        let device = Device::system_default().expect("❌ No Metal device found");
        let queue = device.new_command_queue();

        let metallib_path =
            format!("{}/shaders/kernels.metallib", env!("CARGO_MANIFEST_DIR"));
        let library = device
            .new_library_with_file(&metallib_path)
            .expect("❌ Failed to load Metal library");

        println!("🚀 GPU thread initialized — Metal context bound to this thread");

        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                GpuCommand::DispatchKernel {
                    pipeline_name,
                    buffer_count,
                    reply,
                } => {
                    // Try to look up and build the compute pipeline dynamically
                    let func = match library.get_function(&pipeline_name, None) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ = reply.send(Err(format!(
                                "⚠️ GPU: kernel '{}' not found ({:?})",
                                pipeline_name, e
                            )));
                            continue;
                        }
                    };

                    let pipeline =
                        match device.new_compute_pipeline_state_with_function(&func) {
                            Ok(p) => p,
                            Err(e) => {
                                let _ = reply.send(Err(format!(
                                    "⚠️ GPU: pipeline build failed for '{}' ({:?})",
                                    pipeline_name, e
                                )));
                                continue;
                            }
                        };

                    // Build command buffer and encoder
                    let cmd_buf = queue.new_command_buffer();
                    let encoder = cmd_buf.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pipeline);

                    // Allocate transient buffers (for example/demo)
                    for i in 0..buffer_count {
                        let buf = device.new_buffer(
                            4096,
                            MTLResourceOptions::StorageModeShared,
                        );
                        encoder.set_buffer(i as u64, Some(&buf), 0);
                    }

                    // Dispatch minimal work for demonstration
                    let threads_per_group = pipeline.thread_execution_width() as u64;
                    let total_threads = 1024u64;
                    let tg_count = MTLSize {
                        width: (total_threads + threads_per_group - 1) / threads_per_group,
                        height: 1,
                        depth: 1,
                    };
                    let tg_size = MTLSize {
                        width: threads_per_group,
                        height: 1,
                        depth: 1,
                    };

                    encoder.dispatch_thread_groups(tg_count, tg_size);
                    encoder.end_encoding();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();

                    println!("✅ GPU kernel '{}' completed", pipeline_name);
                    let _ = reply.send(Ok(()));
                }
            }
        }

        println!("🛑 GPU thread exiting — channel closed");
    });

    tx
}

pub fn aligned_u32_buffer(device: &Device, len: usize) -> Buffer {
    device.new_buffer(
        (len * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
    
pub fn aligned_f32_buffer(device: &Device, len: usize) -> Buffer {
    device.new_buffer(
        (len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

