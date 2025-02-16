import asyncio
import json
import time
import argparse
from vllm_benchmark import run_benchmark

async def run_all_benchmarks(vllm_url, api_key, use_long_context, node_number):
    configurations = [
        {"num_requests": 60, "concurrency": 1, "output_tokens": 100},
        {"num_requests": 60, "concurrency": 2, "output_tokens": 100},
        {"num_requests": 60, "concurrency": 3, "output_tokens": 100},
        {"num_requests": 60, "concurrency": 4, "output_tokens": 100},
        {"num_requests": 60, "concurrency": 5, "output_tokens": 100},
        {"num_requests": 60, "concurrency": 6, "output_tokens": 100},
    ]

    all_results = []

    timeout=300
    for config in configurations:
        print(f"Running benchmark with concurrency {config['concurrency']}...")
        results = await run_benchmark(config['num_requests'], config['concurrency'], timeout, config['output_tokens'], vllm_url, api_key, use_long_context, node_number)
        all_results.append(results)
        time.sleep(5)  # Wait a bit between runs to let the system cool down

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--node_number", type=int, default=1, help="Number of Inference Node (default: 1)")
    parser.add_argument("--api_key", type=str, required=False, default='aws', help="API key for vLLM server")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    args = parser.parse_args()

    all_results = asyncio.run(run_all_benchmarks(args.vllm_url, args.api_key, args.use_long_context, args.node_number))

    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Benchmark results saved to benchmark_results.json")

if __name__ == "__main__":
    main()
