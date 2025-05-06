import time
import argparse
import torch
import torch_neuronx

def create_tokens(batch_size, seq_length=2048):
    # Create a single sequence: [0] + 2046 copies of 32 + [2]
    base_tokens = [0] + [32] * (seq_length - 2) + [2]
    # Repeat for the given batch size
    tokens = torch.tensor([base_tokens] * batch_size)
    return tokens

def benchmark(model, tokens, num_runs=10):
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(tokens)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    return avg_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark Torch-NeuronX model inference")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--warmup_runs", type=int, default=10,
                        help="Number of warmup inferences (default: 10)")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                        help="Number of benchmark inferences (default: 10)")
    args = parser.parse_args()
    # model_path = "/home/ubuntu/esmc-300m-inferentia.pt" 
    model_path =  f"/home/ubuntu/esmc-300m-inferentia_batch_size{args.batch_size}.pt"
    # Load the model
    model = torch.jit.load(model_path)
    torch_neuronx.move_trace_to_device(model, 0)

    # Create tokens for the specified batch size
    tokens = create_tokens(args.batch_size)

    # Warmup: run a number of inferences and discard their timings
    with torch.no_grad():
        for _ in range(args.warmup_runs):
            _ = model(tokens)

    # Benchmark: measure inference times
    avg_time = benchmark(model, tokens, args.benchmark_runs)
    print(f"Average inference time over {args.benchmark_runs} runs (batch size {args.batch_size}): {avg_time:.6f} seconds")

if __name__ == "__main__":
    main()
