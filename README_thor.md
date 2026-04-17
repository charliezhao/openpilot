# OpenPilot Jetson Thor Inference & Benchmarking

This directory contains the scripts necessary to evaluate and benchmark the comma.ai OpenPilot models on the Jetson Thor architecture, as discussed. We are targeting high-performance execution using TensorRT and testing reasoning latency alongside driving accuracy.

## Prerequisites for Jetson Thor
Ensure your Thor is set up with the standard NVIDIA JetPack libraries and OpenPilot's basic dependencies:

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev tensorrt libnvinfer-dev libnvonnxparsers-dev

# 2. Install PyCUDA for Zero-Copy memory
pip3 install pycuda numpy onnx
```

## Tools Provided

### 1. `thor_benchmark.py` (Latency, Jitter, & TRT Compilation)
This script handles the compilation of OpenPilot's ONNX models into optimized TensorRT engines using FP16 precision. It heavily utilizes `pycuda` for page-locked (pinned) zero-copy memory allocations to minimize host-to-device memory transfer overhead, crucial for the massive memory pools on Thor.

**Usage:**
```bash
# Compile and benchmark driving_vision.onnx
python3 thor_benchmark.py --onnx selfdrive/modeld/models/driving_vision.onnx --engine driving_vision.trt --runs 500

# Compile and benchmark driving_policy.onnx
python3 thor_benchmark.py --onnx selfdrive/modeld/models/driving_policy.onnx --engine driving_policy.trt --runs 500
```
*Tip:* To measure power (Joules per Inference), run `jtop` simultaneously during the benchmark runs.

### 2. `minade_eval.py` (Accuracy, minADE, FDE)
This script contains the core logic for parsing the model's trajectory output and computing the Minimum Average Displacement Error (minADE) and Final Displacement Error (FDE). 

**Usage:**
```bash
python3 minade_eval.py
```
*Note:* The script is currently mocked. To use it with the real `comma2k19` dataset or Replay logs, feed the model's parsed plan output from `thor_benchmark.py` against the `modelV2` cereal messages in the replay data.

## Shadow Testing (Alpamayo vs. Standard)
To perform shadow testing:
1. Run `thor_benchmark.py` for the standard openpilot vision model. Note the P99 latency.
2. Integrate the VLA model (e.g. Alpamayo) directly onto the Thor in a separate container/process.
3. Feed the same datasets (or live CSI-2 feeds if you have a camera rig) into both concurrently and compare the delay deltas. The difference gives the exact "latency tax" of the larger reasoning model.
