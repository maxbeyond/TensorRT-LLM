import argparse
import math
import torch
import numpy as np
import time
import tensorrt_llm
from tensorrt_llm import Tensor, Builder
from tensorrt_llm._utils import torch_to_numpy, str_dtype_to_np, str_dtype_to_torch
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils, RotaryScalingType
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from transformers import LlamaConfig
import tabulate
from typing import List, Dict, Tuple, Optional, Any, Callable
from itertools import product


class CUDAGraph:
    def __init__(self, graph):
        self.graph = graph

    def capture_begin(self, pool=None, capture_error_mode="global"):
        assert capture_error_mode == "global"
        if pool is None:
            self.graph.capture_begin(0, 0)
        else:
            self.graph.capture_begin(*pool)

    def capture_end(self):
        self.graph.capture_end()

    def replay(self):
        self.graph.replay()

    def reset(self):
        self.graph.reset()

    def release_graph(self):
        self.graph.release_graph()

    def debug_dump(self):
        self.graph.debug_dump()


class GraphedCallable:
    def __init__(
        self,
        graph: CUDAGraph,
        static_inputs: Tuple[torch.Tensor],
        static_outputs: Tuple[torch.Tensor],
    ):
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs

    def __call__(self, *args):
        for i, (static_input, arg) in enumerate(zip(self.static_inputs, args)):
            # A bit of a hack: if the pointer is the same we allow shape to
            # change. It's used for KVCache address reservations.
            if static_input.data_ptr() != arg.data_ptr():
                # copy_ can broadcast so do an explicit check
                assert static_input.shape == arg.shape, f"Input {i} mismatch: {static_input.shape} != {arg.shape}"
                static_input.copy_(arg)
        self.graph.replay()
        return self.static_outputs

    def release_graph(self):
        self.graph.release_graph()


@torch.no_grad()
def my_make_graphed_callable(
    callable: Callable,
    inputs: Tuple[torch.Tensor],
    pool=None,
    num_warmup_iters=3,
) -> GraphedCallable:
    """
    Similar to torch.cuda.make_graphed_callables, but for inference workloads.
    """

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    # with torch.cuda.stream(torch.cuda.Stream()):
    for _ in range(num_warmup_iters):
        callable(*inputs)
    torch.cuda.synchronize()

    # Capture.
    graph = CUDAGraph(torch.classes.firecuda.CUDAGraph())
    with torch.cuda.graph(graph, pool=pool):
        static_outputs = callable(*inputs)

    return GraphedCallable(graph, inputs, static_outputs)


def run_benchmark(bench_func, loop: int = 10, cudagraph: bool = True, reset_inputs=None) -> int:
    """Run a benchmark function with optional CUDA graph optimization."""
    if loop <= 0:
        # Just run once
        loop = 1

    if loop > 1:

        def _loop_func():
            for _ in range(loop):
                bench_func()

        loop_bench_func = _loop_func
    else:
        loop_bench_func = bench_func

    if cudagraph:
        # Make the function a CUDA graph to reduce runtime overhead
        loop_bench_func = my_make_graphed_callable(loop_bench_func, [], num_warmup_iters=1)

    # Warmup
    if loop > 0:
        if reset_inputs is not None:
            reset_inputs()
        loop_bench_func()
        if reset_inputs is not None:
            reset_inputs()

    # Run benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    loop_bench_func()
    end.record()

    # Synchronize and get elapsed time
    end.synchronize()
    elapsed_us = int(start.elapsed_time(end) * 1e3) // loop  # microseconds per iteration

    return elapsed_us


def benchmark_gpt_attention(
    batch_sizes: List[int],
    seq_lengths: List[int],
    num_heads: List[int],
    head_sizes: List[int],
    num_kv_heads: List[int] = None,
    dtypes: List[str] = ["float16"],
    kv_dtypes: List[str] = None,
    context_fmha_types: List[ContextFMHAType] = [ContextFMHAType.disabled],
    paged_kv_cache: bool = False,
    remove_input_padding: bool = False,
    enable_cudagraph: bool = False,
    attention_type: str = "llama_attention",
    loop: int = 10,
    beam_width: int = 1,
    tokens_per_block: int = 128,
):
    """
    Benchmark GPT attention using TensorRT-LLM's _construct_execution

    Args:
        batch_sizes: List of batch sizes to benchmark
        seq_lengths: List of sequence lengths to benchmark
        num_heads: List of attention head counts to benchmark
        head_sizes: List of head dimensions to benchmark
        num_kv_heads: List of KV head counts for MQA/GQA, defaults to num_heads if None
        dtypes: List of data types to benchmark
        kv_dtypes: List of KV cache data types, defaults to dtype if None
        context_fmha_types: List of FMHA types to benchmark
        paged_kv_cache: Whether to use paged KV cache
        remove_input_padding: Whether to remove input padding
        enable_cudagraph: Whether to use CUDA graph for benchmarking
        attention_type: Type of attention to benchmark
        loop: Number of iterations for benchmarking
        beam_width: Beam width for the benchmark
        tokens_per_block: Number of tokens per block for paged KV cache

    Returns:
        List of benchmark results
    """
    torch.random.manual_seed(0)
    device = torch.device("cuda")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    if kv_dtypes is None:
        kv_dtypes = dtypes

    results = [
        [
            "batch_size",
            "seq_len",
            "num_heads",
            "head_size",
            "num_kv_heads",
            "dtype",
            "kv_dtype",
            "fmha_type",
            "paged_kv",
            "time_us",
            "tflops",
            "mem_bw_gb/s",
        ]
    ]

    print("input:")
    print("batch_sizes:", batch_sizes)
    print("seq_lengths:", seq_lengths)
    print("num_heads:", num_heads)
    print("head_sizes:", head_sizes)
    print("num_kv_heads:", num_kv_heads)
    print("dtypes:", dtypes)
    print("kv_dtypes:", kv_dtypes)
    print("context_fmha_type", context_fmha_types)
    print("paged_kv_cache:", paged_kv_cache)
    print("remove_input_padding:", remove_input_padding)
    print("enable_cudagraph:", enable_cudagraph)
    print("attention_type:", attention_type)
    print("loop:", loop)
    print("beam_width:", beam_width)
    print("tokens_per_block:", tokens_per_block)

    for batch_size, seq_len, n_heads, head_size, n_kv_heads, dtype, kv_dtype, context_fmha_type in product(
        batch_sizes,
        seq_lengths,
        num_heads,
        head_sizes,
        num_kv_heads if isinstance(num_kv_heads, list) else [n_heads for n_heads in num_heads],
        dtypes,
        kv_dtypes,
        context_fmha_types,
    ):
        hidden_size = n_heads * head_size
        kv_hidden_size = n_kv_heads * head_size

        # Determine if using quantized KV cache
        use_int8_kv_cache = kv_dtype == "int8"
        use_fp8_kv_cache = kv_dtype == "fp8"

        # Set up the benchmark tensors
        max_seq_len = seq_len + 24  # Add some buffer for generation
        max_blocks_per_seq = math.ceil(max_seq_len / tokens_per_block)
        num_blocks = batch_size * beam_width * max_blocks_per_seq

        # Create configuration
        config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_seq_len,
            torch_dtype=dtype,
            rope_theta=10000.0,
        )

        # Create input tensors
        input_tensor = (
            torch.randn(
                (batch_size, seq_len, hidden_size),
                dtype=str_dtype_to_torch(dtype),
                device=device,
            )
            * 1e-3
        )

        if remove_input_padding:
            input_tensor = input_tensor.reshape(batch_size * seq_len, hidden_size)

        # QKV projection weight and bias
        plugin_kv_num_heads = n_kv_heads if attention_type == "llama_attention" else n_heads
        qkv_hidden_size = hidden_size + 2 * plugin_kv_num_heads * head_size

        weight = (
            torch.randn(
                (hidden_size, qkv_hidden_size),
                dtype=str_dtype_to_torch(dtype),
                device=device,
            )
            * 1e-3
        )

        bias = torch.randn((qkv_hidden_size,), dtype=str_dtype_to_torch(dtype), device=device) * 1e-2

        # KV cache tensor
        torch_kv_cache_dtype = str_dtype_to_torch("int8") if kv_dtype == "fp8" else str_dtype_to_torch(kv_dtype)

        if paged_kv_cache:
            past_key_value_shape = (
                num_blocks,
                2,
                plugin_kv_num_heads,
                tokens_per_block,
                head_size,
            )
        else:
            past_key_value_shape = (
                batch_size,
                2,
                plugin_kv_num_heads,
                max_seq_len,
                head_size,
            )

        past_key_value = torch.zeros(past_key_value_shape, dtype=torch_kv_cache_dtype, device=device)

        # Other required tensors
        input_lengths = torch.ones((batch_size,), dtype=torch.int32, device=device) * seq_len
        host_context_lengths = input_lengths.cpu() if remove_input_padding else None
        sequence_length = input_lengths.clone()
        host_past_key_value_lengths = torch.zeros_like(input_lengths, device="cpu")
        host_max_attention_window_sizes = torch.tensor([max_seq_len], dtype=torch.int32)
        host_sink_token_length = torch.tensor([0], dtype=torch.int32)
        host_request_types = torch.zeros((batch_size,), dtype=torch.int32)

        cache_indirection = torch.full((batch_size, beam_width, max_seq_len), 0, dtype=torch.int32, device=device)

        # KV quantization scales
        kv_quant_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        kv_dequant_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        # Host perf knobs and progress tensor
        host_runtime_perf_knobs = torch.tensor([-1] * 16, dtype=torch.int64, device="cpu")
        host_runtime_perf_knobs[0] = 1  # Enable multi-block mode
        if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
            host_runtime_perf_knobs[1] = 1  # Enable FP32 accumulation

        host_context_progress = torch.tensor([0], dtype=torch.int64, device="cpu")

        # Output tensor
        output_shape = input_tensor.shape
        output = torch.zeros(output_shape, dtype=str_dtype_to_torch(dtype), device=device)

        # Paged KV cache setup
        host_kv_cache_block_offsets = None
        host_kv_cache_pool_pointers = None
        host_kv_cache_pool_mapping = None

        if paged_kv_cache:
            from tensorrt_llm.runtime.memory_pools.memory_pools_allocator import (
                MemoryPoolsAllocator,
            )
            from tensorrt_llm.runtime.memory_pools.pools_kv_cache_manager import (
                PoolsKVCacheManager,
            )

            # Set up memory pool allocator
            memory_pools_allocator = MemoryPoolsAllocator(
                num_blocks=num_blocks,
                tokens_per_block=tokens_per_block,
                head_size=head_size,
            )

            num_kv_heads_per_layer = MemoryPoolsAllocator.prepare_num_kv_heads_per_layer(plugin_kv_num_heads, 1)
            memory_pools_allocator.allocate(dtype, num_kv_heads_per_layer)

            # Set up KV cache manager
            pools_kv_cache_manager = PoolsKVCacheManager(
                memory_pools_allocator.pools_metadata,
                max_blocks_per_seq,
                num_blocks,
                tokens_per_block,
                head_size,
                max_attention_window_size=max_seq_len,
                beam_width=beam_width,
                sink_token_len=0,
            )

            # Add sequences to KV cache manager
            from tensorrt_llm.runtime import GenerationSequence

            for bi in range(batch_size):
                pools_kv_cache_manager.add_sequence(GenerationSequence(seq_idx=bi, batch_idx=bi), seq_len)

            # Get block offsets
            kv_cache_manager = pools_kv_cache_manager.get_single_kv_cache_manager()
            host_kv_cache_block_offsets = kv_cache_manager.get_block_offsets(beam_width)

            # Configure pool pointers
            host_kv_cache_pool_pointers = torch.tensor([past_key_value.data_ptr(), 0], dtype=torch.int64)
            host_kv_cache_pool_mapping = memory_pools_allocator.pool_mapping

        # Wrap in callable for benchmarking
        position_embedding_type = PositionEmbeddingType.rope_gpt_neox if attention_type == "llama_attention" else None

        # Initialize context
        shape_dict = {
            "weight": (hidden_size, qkv_hidden_size),
            "bias": (qkv_hidden_size,),
            "host_past_key_value_lengths": (batch_size,),
            "host_max_attention_window_sizes": (1,),
            "host_sink_token_length": (1,),
            "sequence_length": (batch_size,),
            "context_lengths": (batch_size,),
            "kv_quant_scale": (1,),
            "kv_dequant_scale": (1,),
            "cache_indirection": (batch_size, beam_width, max_seq_len),
            "host_request_types": (batch_size,),
        }

        streamingllm = False
        enable_remove_input_padding = False
        kv_cache_dtype = "float16"
        fuse_bias = False

        def _construct_execution(
            session,
            input_tensor,
            weight,
            bias,
            past_key_value,
            host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            attention_packed_mask,
            sequence_length,
            host_past_key_value_lengths,
            host_max_attention_window_sizes,
            host_sink_token_length,
            context_lengths,
            host_context_lengths,
            cache_indirection,
            host_request_types,
            num_heads,
            hidden_size,
            num_kv_heads,
            output,
            dtype,
            position_embedding_type,
            max_context_length,
            shape_dict,
            kv_int8_quant_scale,
            kv_int8_dequant_scale,
            configuration,
            host_runtime_perf_knobs,
            host_context_progress,
        ):
            kv_cache_block_offsets = None
            if paged_kv_cache:
                kv_cache_block_offsets = host_kv_cache_block_offsets.to("cuda")
            head_size = hidden_size // num_heads
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.gpt_attention_plugin = dtype
            net.plugin_config.set_context_fmha(context_fmha_type)
            net.plugin_config.use_fp8_context_fmha = False
            net.plugin_config.use_paged_context_fmha = False
            if streamingllm:
                net.plugin_config.streamingllm = True
            if enable_remove_input_padding:
                net.plugin_config.remove_input_padding = True
            else:
                net.plugin_config.remove_input_padding = False
            if paged_kv_cache:
                net.plugin_config.enable_paged_kv_cache(tokens_per_block)
            else:
                net.plugin_config.paged_kv_cache = False

            with tensorrt_llm.net_guard(net):
                x_tensor = Tensor(
                    name="input", shape=tuple(input_tensor.shape), dtype=tensorrt_llm.str_dtype_to_trt(dtype)
                )
                attention_packed_mask_tensor = None
                if attention_packed_mask is not None:
                    attention_packed_mask_tensor = Tensor(
                        name="attention_packed_mask",
                        shape=tuple(attention_packed_mask.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                    )
                sequence_length_tensor = Tensor(
                    name="sequence_length",
                    shape=tuple(sequence_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                )
                host_past_key_value_lengths_tensor = Tensor(
                    name="host_past_key_value_lengths",
                    shape=tuple(host_past_key_value_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                )
                host_max_attention_window_sizes_tensor = Tensor(
                    name="host_max_attention_window_sizes",
                    shape=tuple(host_max_attention_window_sizes.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                )
                host_sink_token_length_tensor = Tensor(
                    name="host_sink_token_length",
                    shape=tuple(host_sink_token_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                )
                context_lengths_tensor = Tensor(
                    name="context_lengths",
                    shape=tuple(context_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                )
                host_context_lengths_tensor = (
                    Tensor(
                        name="host_context_lengths",
                        shape=tuple(context_lengths.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                    )
                    if enable_remove_input_padding
                    else None
                )
                cache_indirection_tensor = Tensor(
                    name="cache_indirection",
                    shape=tuple(cache_indirection.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                )
                host_request_types_tensor = Tensor(
                    name="host_request_types",
                    shape=tuple(host_request_types.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                )
                host_runtime_perf_knobs_tensor = Tensor(
                    name="host_runtime_perf_knobs", shape=[16], dtype=tensorrt_llm.str_dtype_to_trt("int64")
                )
                host_context_progress_tensor = Tensor(
                    name="host_context_progress", shape=[1], dtype=tensorrt_llm.str_dtype_to_trt("int64")
                )

                past_key_value_tensor = None
                kv_cache_block_offsets_tensor = None
                host_kv_cache_block_offsets_tensor = None
                host_kv_cache_pool_pointers_tensor = None
                host_kv_cache_pool_mapping_tensor = None
                if paged_kv_cache:
                    kv_cache_block_offsets_tensor = Tensor(
                        name="kv_cache_block_offsets",
                        shape=tuple(kv_cache_block_offsets.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                    )
                    host_kv_cache_block_offsets_tensor = Tensor(
                        name="host_kv_cache_block_offsets",
                        shape=tuple(kv_cache_block_offsets.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt("int32"),
                    )
                    host_kv_cache_pool_pointers_tensor = Tensor(
                        name="host_kv_cache_pool_pointers",
                        shape=(
                            1,
                            1,
                        ),
                        dtype=tensorrt_llm.str_dtype_to_trt("int64"),
                    )
                    host_kv_cache_pool_mapping_tensor = Tensor(
                        name="host_kv_cache_pool_mapping", shape=(1, 1), dtype=tensorrt_llm.str_dtype_to_trt("int32")
                    )
                else:
                    past_key_value_tensor = Tensor(
                        name="past_key_value",
                        shape=tuple(past_key_value.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype),
                    )

                kv_quant_scale_tensor = None
                kv_dequant_scale_tensor = None
                if use_int8_kv_cache or use_fp8_kv_cache:
                    kv_quant_scale_tensor = Tensor(
                        name="kv_quant_scale", shape=(1,), dtype=tensorrt_llm.str_dtype_to_trt("float32")
                    )
                    kv_dequant_scale_tensor = Tensor(
                        name="kv_dequant_scale", shape=(1,), dtype=tensorrt_llm.str_dtype_to_trt("float32")
                    )

                linear = tensorrt_llm.layers.Linear(
                    hidden_size,
                    weight.size()[-1],
                    bias=attention_type in ["gpt2_attention", "llama_attention", "gpt_bigcode_attention"],
                )
                linear.weight.value = np.ascontiguousarray(torch_to_numpy(weight.T.cpu()))
                if attention_type in ["gpt2_attention", "llama_attention", "gpt_bigcode_attention"]:
                    linear.bias.value = torch_to_numpy(bias.cpu())

                if fuse_bias:
                    qkv = tensorrt_llm.functional.matmul(x_tensor, linear.weight.value, transb=True)
                    qkv_bias = (
                        tensorrt_llm.functional.constant(
                            np.zeros((linear.out_features,), dtype=str_dtype_to_np(dtype))
                        )
                        if linear.bias is None
                        else linear.bias.value
                    )
                else:
                    qkv = linear(x_tensor)
                    qkv_bias = None

                rotary_embedding_dim = head_size if attention_type in ["llama_attention", "gptj_attention"] else 0
                if position_embedding_type is None:
                    # If the caller doesn't specify position_embedding_type explicitly, infer it from attention_type.
                    if attention_type == "llama_attention":
                        position_embedding_type = PositionEmbeddingType.rope_gpt_neox
                    elif attention_type == "gptj_attention":
                        position_embedding_type = PositionEmbeddingType.rope_gptj
                    else:
                        position_embedding_type = PositionEmbeddingType.learned_absolute

                rope_base = 10000.0
                rope_scale_type = RotaryScalingType.none
                rope_scale = 1.0
                if attention_type == "llama_attention":
                    rope_base = configuration.rope_theta
                    if configuration.rope_scaling is not None:
                        rope_scale_type = {"linear": RotaryScalingType.linear, "dynamic": RotaryScalingType.dynamic}[
                            configuration.rope_scaling["type"]
                        ]
                        rope_scale = configuration.rope_scaling["factor"]
                rotary_inv_freq, embed_positions_for_gpt_attention = (
                    RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                        configuration.max_position_embeddings, rotary_embedding_dim, rope_base, rope_scale
                    )
                )
                rotary_inv_freq_cache = (
                    tensorrt_llm.functional.constant(rotary_inv_freq) if position_embedding_type.is_rope() else None
                )
                rotary_cos_sin = (
                    tensorrt_llm.functional.constant(embed_positions_for_gpt_attention)
                    if position_embedding_type.is_rope()
                    else None
                )

                mrope_rotary_cos_sin = (
                    tensorrt_llm.functional.constant(embed_positions_for_gpt_attention)
                    if position_embedding_type.is_mrope()
                    else None
                )
                mrope_position_deltas = sequence_length_tensor

                outputs = tensorrt_llm.functional.gpt_attention(
                    qkv=qkv,
                    attention_packed_mask=attention_packed_mask_tensor,
                    past_key_value=past_key_value_tensor,
                    sequence_length=sequence_length_tensor,
                    host_past_key_value_lengths=host_past_key_value_lengths_tensor,
                    host_max_attention_window_sizes=host_max_attention_window_sizes_tensor,
                    host_sink_token_length=host_sink_token_length_tensor,
                    context_lengths=context_lengths_tensor,
                    cache_indirection=cache_indirection_tensor,
                    host_request_types=host_request_types_tensor,
                    layer_idx=0,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    hidden_size_per_head=head_size,
                    q_scaling=1.0,
                    rotary_embedding_dim=rotary_embedding_dim,
                    rotary_embedding_base=rope_base,
                    rotary_embedding_scale_type=rope_scale_type,
                    rotary_embedding_scale=rope_scale,
                    rotary_embedding_max_positions=configuration.max_position_embeddings,
                    position_embedding_type=position_embedding_type,
                    rotary_inv_freq=rotary_inv_freq_cache,
                    rotary_cos_sin=rotary_cos_sin,
                    kv_orig_quant_scale=kv_quant_scale_tensor,
                    kv_quant_orig_scale=kv_dequant_scale_tensor,
                    host_context_lengths=host_context_lengths_tensor,
                    kv_cache_quant_mode=QuantMode.from_description(
                        use_int8_kv_cache=use_int8_kv_cache, use_fp8_kv_cache=use_fp8_kv_cache
                    ),
                    kv_cache_block_offsets=kv_cache_block_offsets_tensor,
                    host_kv_cache_block_offsets=host_kv_cache_block_offsets_tensor,
                    host_kv_cache_pool_pointers=host_kv_cache_pool_pointers_tensor,
                    host_kv_cache_pool_mapping=host_kv_cache_pool_mapping_tensor,
                    max_context_length=max_context_length,
                    qkv_bias=qkv_bias,
                    mrope_rotary_cos_sin=mrope_rotary_cos_sin,
                    mrope_position_deltas=mrope_position_deltas,
                    host_runtime_perf_knobs=host_runtime_perf_knobs_tensor,
                    host_context_progress=host_context_progress_tensor,
                )

                net._mark_output(outputs[0], "output", dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                if not paged_kv_cache:
                    net._mark_output(
                        outputs[1], "present_key_value", dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype)
                    )

            inputs = {
                "input": input_tensor,
                "sequence_length": sequence_length,
                "host_past_key_value_lengths": host_past_key_value_lengths,
                "host_max_attention_window_sizes": host_max_attention_window_sizes,
                "host_sink_token_length": host_sink_token_length,
                "context_lengths": context_lengths,
                "cache_indirection": cache_indirection,
                "host_request_types": host_request_types,
                "host_runtime_perf_knobs": host_runtime_perf_knobs,
                "host_context_progress": host_context_progress,
            }
            if attention_packed_mask is not None:
                inputs["attention_packed_mask"] = attention_packed_mask
            if paged_kv_cache:
                inputs["kv_cache_block_offsets"] = kv_cache_block_offsets
                inputs["host_kv_cache_block_offsets"] = host_kv_cache_block_offsets
                inputs["host_kv_cache_pool_pointers"] = host_kv_cache_pool_pointers
                inputs["host_kv_cache_pool_mapping"] = host_kv_cache_pool_mapping
            else:
                inputs["past_key_value"] = past_key_value

            if use_int8_kv_cache or use_fp8_kv_cache:
                inputs["kv_quant_scale"] = kv_quant_scale
                inputs["kv_dequant_scale"] = kv_dequant_scale

            if enable_remove_input_padding:
                inputs["host_context_lengths"] = host_context_lengths

            outputs = {"output": output}
            if not paged_kv_cache:
                outputs["present_key_value"] = past_key_value

            stream = torch.cuda.current_stream()
            # NOTE: when 8-bit kv cache is used together with paged kv cache no 8-bit tensors are exposed to TRT
            int8_trt_flag = use_int8_kv_cache and not paged_kv_cache
            use_fp8_kv_cache and not paged_kv_cache
            quant_mode = (
                QuantMode.from_description(use_fp8_kv_cache=use_fp8_kv_cache)
                if use_fp8_kv_cache and not paged_kv_cache
                else QuantMode(0)
            )
            builder_config = builder.create_builder_config(
                name=attention_type, precision=dtype, int8=int8_trt_flag, quant_mode=quant_mode
            )

            if session is None:
                print("dbg build engine")
                engine = builder.build_engine(net, builder_config)
                session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
            session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)

            torch.cuda.synchronize()
            return session, outputs["output"], past_key_value

        # Create a context/session once for warm-up
        session, _, _ = _construct_execution(
            None,
            input_tensor,
            weight,
            bias,
            past_key_value,
            host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            None,
            sequence_length,
            host_past_key_value_lengths,
            host_max_attention_window_sizes,
            host_sink_token_length,
            input_lengths,
            host_context_lengths,
            cache_indirection,
            host_request_types,
            n_heads,
            hidden_size,
            n_kv_heads,
            output,
            dtype,
            position_embedding_type,
            seq_len,
            shape_dict,
            kv_quant_scale,
            kv_dequant_scale,
            config,
            host_runtime_perf_knobs,
            host_context_progress,
        )

        # Create benchmark function
        def bench_func():
            nonlocal session, output
            session, _, _ = _construct_execution(
                session,
                input_tensor,
                weight,
                bias,
                past_key_value,
                host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping,
                None,
                sequence_length,
                host_past_key_value_lengths,
                host_max_attention_window_sizes,
                host_sink_token_length,
                input_lengths,
                host_context_lengths,
                cache_indirection,
                host_request_types,
                n_heads,
                hidden_size,
                n_kv_heads,
                output,
                dtype,
                position_embedding_type,
                seq_len,
                shape_dict,
                kv_quant_scale,
                kv_dequant_scale,
                config,
                host_runtime_perf_knobs,
                host_context_progress,
            )

        # for i in range(10):
        #     bench_func()

        # Run benchmark
        time_us = run_benchmark(bench_func, loop=loop, cudagraph=False)

        # Calculate FLOPs:
        # - Each head computes: Q*K^T (hdim*seq_len*seq_len), softmax (negligible), V (seq_len*seq_len*hdim)
        # - Total: 2 * seq_len * seq_len * hdim per head
        flops = 2 * seq_len * seq_len * head_size
        # Multiply by the number of attention heads
        flops *= n_heads
        # Multiply by batch size
        flops *= batch_size
        # Convert to TFLOPS
        tflops = flops / (time_us * 1e-6) / 1e12

        # Calculate memory bandwidth (KV cache read/write)
        # Memory for KV cache: 2 * n_kv_heads * seq_len * head_size * element_size
        element_size = 2  # bytes for FP16/BF16
        if kv_dtype == "int8" or kv_dtype == "fp8":
            element_size = 1
        elif kv_dtype == "float32":
            element_size = 4

        mem_bytes = 2 * n_kv_heads * seq_len * head_size * element_size * batch_size
        mem_bw_gbs = mem_bytes / (time_us * 1e-6) / 1e9

        # Add result
        results.append(
            [
                batch_size,
                seq_len,
                n_heads,
                head_size,
                n_kv_heads,
                dtype,
                kv_dtype,
                str(context_fmha_type).split(".")[-1],
                paged_kv_cache,
                time_us,
                f"{tflops:.2f}",
                f"{mem_bw_gbs:.2f}",
            ]
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GPT Attention")

    # Model parameters
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32], help="Batch sizes to benchmark")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[640], help="Sequence lengths to benchmark")
    parser.add_argument("--num_heads", type=int, nargs="+", default=[64], help="Number of attention heads")
    parser.add_argument("--head_sizes", type=int, nargs="+", default=[128], help="Size of each attention head")
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        nargs="+",
        default=8,
        help="Number of KV heads for MQA/GQA (defaults to num_heads)",
    )

    # Data type parameters
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["float16"],
        choices=["float16", "bfloat16", "float32"],
        help="Data types to benchmark",
    )
    parser.add_argument(
        "--kv_dtypes",
        type=str,
        nargs="+",
        default=None,
        choices=["float16"],
        help="KV cache data types (defaults to dtypes)",
    )

    # TensorRT parameters
    parser.add_argument(
        "--context_fmha_types",
        type=str,
        nargs="+",
        default=["enabled"],
        choices=["disabled", "enabled", "enabled_with_fp32_acc"],
        help="FMHA types to benchmark",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="llama_attention",
        choices=["llama_attention"],
        help="Type of attention to benchmark",
    )

    # Optimization parameters
    parser.add_argument("--paged_kv_cache", default=True, action="store_true", help="Use paged KV cache")
    parser.add_argument("--remove_input_padding", action="store_true", help="Remove input padding")
    parser.add_argument("--disable_cudagraph", action="store_true", help="Disable CUDA graph")
    parser.add_argument("--loop", type=int, default=5, help="Number of iterations for benchmarking")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width for benchmarking")

    args = parser.parse_args()

    # Convert context_fmha_type strings to enum values
    fmha_type_map = {
        "disabled": ContextFMHAType.disabled,
        "enabled": ContextFMHAType.enabled,
        "enabled_with_fp32_acc": ContextFMHAType.enabled_with_fp32_acc,
    }
    context_fmha_types = [fmha_type_map[t] for t in args.context_fmha_types]

    # Run benchmark
    results = benchmark_gpt_attention(
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_heads=args.num_heads,
        head_sizes=args.head_sizes,
        num_kv_heads=args.num_kv_heads,
        dtypes=args.dtypes,
        kv_dtypes=args.kv_dtypes,
        context_fmha_types=context_fmha_types,
        paged_kv_cache=args.paged_kv_cache,
        remove_input_padding=args.remove_input_padding,
        enable_cudagraph=not args.disable_cudagraph,
        attention_type=args.attention_type,
        loop=args.loop,
        beam_width=args.beam_width,
    )

    # Print results
    print(tabulate.tabulate(results, headers="firstrow"))
