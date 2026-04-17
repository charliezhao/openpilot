import argparse
import time
import numpy as np
import os
import json

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# TRT Logger
if TRT_AVAILABLE:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    
    engine = context.engine
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    
    context.execute_async_v3(stream_handle=stream.handle)
    
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

def build_engine(onnx_file_path, engine_file_path):
    print(f"Building TensorRT engine from {onnx_file_path}...")
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:
        
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Use FP16 for Thor performance
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        plan = builder.build_serialized_network(network, config)
        with open(engine_file_path, 'wb') as f:
            f.write(plan)
        
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(plan)

def load_engine(engine_file_path):
    print(f"Loading TensorRT engine from {engine_file_path}...")
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

def run_benchmark(args, engine=None):
    num_runs = args.runs
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running benchmark ({'MOCK' if args.mock else 'REAL'})...")
    latencies = []
    
    if args.mock:
        # Simulate latencies based on Jetson Thor profile (expected 4-8ms for vision)
        for _ in range(num_runs):
            t0 = time.time()
            time.sleep(np.random.uniform(0.004, 0.008)) # simulate 4-8ms
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)
    else:
        # Use a context manager for the execution context
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            
            # Fill inputs with random data
            for inp in inputs:
                np.copyto(inp.host, np.random.randn(*inp.host.shape).astype(inp.host.dtype))
            
            print("Warming up...")
            for _ in range(10): 
                do_inference(context, bindings, inputs, outputs, stream)
            
            print(f"Executing {num_runs} runs...")
            for _ in range(num_runs):
                t0 = time.time()
                do_inference(context, bindings, inputs, outputs, stream)
                t1 = time.time()
                latencies.append((t1 - t0) * 1000)

    results = {
        "model": args.onnx,
        "runs": num_runs,
        "avg_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mode": "mock" if args.mock else "real"
    }
    
    # Save results to file
    out_file = os.path.join(output_dir, f"inference_{os.path.basename(args.onnx)}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("-" * 30)
    print(f"Benchmark Complete: {results['avg_ms']:.2f}ms avg")
    print(f"Results saved to: {out_file}")
    print("-" * 30)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='selfdrive/modeld/models/driving_vision.onnx')
    parser.add_argument('--engine', type=str, default='driving_vision.trt')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--mock', action='store_true', help='Run in mock mode (no GPU)')
    args = parser.parse_args()
    
    if args.mock:
        run_benchmark(args)
    else:
        if not TRT_AVAILABLE:
            print("TensorRT or PyCUDA not available. Use --mock for verification.")
            exit(1)
        
        engine = None
        if os.path.exists(args.engine):
            engine = load_engine(args.engine)
        else:
            if not os.path.exists(args.onnx):
                print(f"ONNX model not found: {args.onnx}")
                exit(1)
            engine = build_engine(args.onnx, args.engine)
        
        if engine:
            run_benchmark(args, engine)
        else:
            print("Failed to load or build engine.")
            exit(1)
