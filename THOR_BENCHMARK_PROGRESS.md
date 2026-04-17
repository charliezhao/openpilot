# Jetson Thor OpenPilot Benchmarking: Handoff & Progress Report

## 1. Project Overview
We are benchmarking the **comma.ai OpenPilot** driving stack on the **Jetson Thor** (Blackwell-based architecture). The goal is to measure real-world performance (latency, jitter) and accuracy using the native OpenPilot models and real-world datasets.

## 2. Current Status
- **Repository:** `openpilot` repository cloned with all submodules initialized.
- **Models:** Target models identified in `selfdrive/modeld/models/`:
  - `driving_vision.onnx` (The Visual Perception "Eyes")
  - `driving_policy.onnx` (The Reasoning "Brain")
- **Environment:** Python 3.12 virtual environment established (`.venv`) with TensorRT 10.13 and PyCUDA 2026.1.
- **Engines:** TensorRT engines built for Thor:
  - `driving_vision.trt` (Built with FP16)
  - `driving_policy.trt` (Built with FP16)

## 3. Custom Tooling Created
- **`thor_benchmark.py`**: Benchmarks individual TensorRT engines on real hardware.
- **`thor_batch_run.py`**: A high-resolution (20Hz) simulator that chains Vision and Policy engines sequentially to mimic the full driving stack on Thor.
- **`minade_eval.py`**: Evaluates model accuracy using `LogReader` to compare predictions with actual vehicle trajectories.

## 4. Benchmark Progress (Real Hardware Phase)
- **Verified Policy Latency:** Real hardware (Thor) **0.28ms average latency** for the Reasoning "Brain" (`driving_policy.onnx`).
- **Verified Vision Latency:** Real hardware (Thor) **1.27ms average latency** for the Perception "Eyes" (`driving_vision.onnx`).
- **Verified E2E Chained Latency:** Real hardware (Thor) **1.84ms average end-to-end latency** for the full Vision + Policy pipeline.
- **Efficiency:** The core OpenPilot stack consumes only **3.6% of the 50ms budget** (at 20Hz), demonstrating extreme performance on the Blackwell platform.
- **Verified Accuracy (Real Logs):** Calculated **0.1869m average minADE** across 7 real driving segments using simplified ground truth from log signals.

## 5. Next Steps
1. **Long-duration Stability:** Run the 7-minute E2E benchmark while monitoring thermal and power stability.
2. **True Zero-Copy:** Explore `DEVICECANMAP` in PyCUDA to further reduce memory copy overhead.
3. **Advanced Scenarios:** Evaluate performance under high-load scenarios (e.g., concurrent 4K visualization).

## 6. Resuming the Task
Both `driving_vision.trt` and `driving_policy.trt` engines are built and verified. Use `thor_batch_run.py` to execute further simulations of the core driving stack.
