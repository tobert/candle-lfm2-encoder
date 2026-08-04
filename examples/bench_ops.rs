//! Where does a forward pass actually spend its time? Times the two
//! operator kinds in isolation against a plain matmul baseline.
//!
//! ```text
//! cargo run --release --example bench_ops
//! ```

use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig};

const HIDDEN: usize = 1024;
const FFN: usize = 4608;
const REPS: usize = 20;

fn bench(label: &str, reps: usize, mut f: impl FnMut()) {
    f(); // warm
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    let per = t.elapsed() / reps as u32;
    println!("  {label:<44} {per:>12.2?}");
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!("candle CPU, f32\n");

    for seq in [12usize, 128, 512] {
        println!("seq = {seq}");

        let x = Tensor::randn(0f32, 1., (1, seq, HIDDEN), &dev)?;
        let w = Tensor::randn(0f32, 1., (FFN, HIDDEN), &dev)?;
        bench("linear (1024 -> 4608) matmul", REPS, || {
            let _ = x.broadcast_matmul(&w.t().unwrap()).unwrap();
        });

        // Depthwise short conv, exactly as the trunk builds it.
        let bx = Tensor::randn(0f32, 1., (1, HIDDEN, seq), &dev)?;
        let cw = Tensor::randn(0f32, 1., (HIDDEN, 1, 3), &dev)?;
        let depthwise = Conv1d::new(
            cw.clone(),
            None,
            Conv1dConfig {
                padding: 1,
                groups: HIDDEN,
                ..Default::default()
            },
        );
        bench("depthwise conv1d (groups = 1024)", REPS, || {
            let _ = depthwise.forward(&bx).unwrap();
        });

        // The same math expressed as 3 shifted multiply-adds instead of a
        // grouped convolution.
        bench("shift-multiply-add equivalent", REPS, || {
            let mut acc: Option<Tensor> = None;
            for k in 0..3usize {
                let tap = cw.narrow(2, k, 1).unwrap().reshape((1, HIDDEN, 1)).unwrap();
                let shifted = bx.narrow(2, 0, seq).unwrap();
                let term = shifted.broadcast_mul(&tap).unwrap();
                acc = Some(match acc {
                    None => term,
                    Some(a) => (a + term).unwrap(),
                });
            }
            let _ = acc.unwrap();
        });

        let dt = Instant::now();
        let _ = Tensor::randn(0f32, 1., (1, 16, seq, 64), &dev)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        println!("  {:<44} {:>12.2?}", "(alloc reference)", dt.elapsed());
        println!();
    }
    Ok(())
}
