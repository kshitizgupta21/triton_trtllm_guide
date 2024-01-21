## Using Triton TensorRT-LLM Backend through building the container

### 1. Get TRTLLM Backend repo
```
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.7.1
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive
```

### 2. Build the Triton TRT-LLM backend container (this will also install `tensorrt_llm` automatically) but building the container can take a while.

```
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
```

### 3. Launch the docker container
```
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v $(pwd):/workspace -w /workspace  triton_trt_llm bash
```

### 4. Build engines
```
cd tensorrt_llm/examples/llama

# Download weights from HuggingFace Transformers
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf

# Build TensorRT engines
export HF_LLAMA_MODEL=$(pwd)/Llama-2-7b-hf/
python build.py --model_dir ${HF_LLAMA_MODEL} \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir /tmp/llama/7B/trt_engines/fp16/4-gpu/ \
                --paged_kv_cache \
                --max_batch_size 64
                --world_size 4 \
                --tp_size 4 \
                --pp_size 1
```

### 5. Create model repository
```
# Create the model repository that will be used by the Triton server
cd /workspace/tensorrtllm_backend
mkdir triton_model_repo

# Copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/ensemble triton_model_repo/
cp -r all_models/inflight_batcher_llm/preprocessing triton_model_repo/
cp -r all_models/inflight_batcher_llm/postprocessing triton_model_repo/
cp -r all_models/inflight_batcher_llm/tensorrt_llm triton_model_repo/

# Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
cp tensorrt_llm/examples/gpt/engines/fp16/4-gpu/* triton_model_repo/tensorrt_llm/1
```

### 6. Modify the model configuration
The following table shows the fields that need to be modified before deployment:

*triton_model_repo/preprocessing/config.pbtxt*

Mandatory ones are
| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 64 |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `${HF_LLAMA_MODEL}`|
| `tokenizer_type` | The type of the tokenizer for the model, `t5`, `auto` and `llama` are supported. In this example, the type should be set to `llama` |
| `preprocessing_instance_count` | Here setting to 1 |

#### Run the following command to prepare preprocessing config.pbtxt
```
python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:llama,triton_max_batch_size:64,preprocessing_instance_count:1
```


*triton_model_repo/tensorrt_llm/config.pbtxt*

Mandatory ones are
| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 64 |
| `max_queue_delay_microseconds` | Here setting to 600 |
| `decoupled` | Controls streaming. Decoupled mode must be set to `True` if using the streaming option from the client. |
| `gpt_model_type` | Set to `inflight_fused_batching` when enabling in-flight batching support. To disable in-flight batching, set to `V1` |
| `gpt_model_path` | Path to the TensorRT-LLM engines for deployment. In this example, the path should be set to `/tmp/llama/7B/trt_engines/fp16/4-gpu/` |

Optional
| Name | Description
| :----------------------: | :-----------------------------: |
| `max_beam_width` | The maximum beam width that any request may ask for when using beam search |
| `max_tokens_in_paged_kv_cache` | The maximum size of the KV cache in number of tokens |
| `max_attention_window_size` | When using techniques like sliding window attention, the maximum number of tokens that are attended to generate one token. Defaults to maximum sequence length |
| `batch_scheduler_policy` | Set to `max_utilization` to greedily pack as many requests as possible in each current in-flight batching iteration. This maximizes the throughput but may result in overheads due to request pause/resume if KV cache limits are reached during execution. Set to `guaranteed_no_evict` to guarantee that a started request is never paused.|
| `kv_cache_free_gpu_mem_fraction` | Set to a number between 0 and 1 to indicate the maximum fraction of GPU memory (after loading the model) that may be used for KV cache|
| `max_num_sequences` | Maximum number of sequences that the in-flight batching scheme can maintain state for. Defaults to `max_batch_size` if `enable_trt_overlap` is `false` and to `2 * max_batch_size` if `enable_trt_overlap` is `true`, where `max_batch_size` is the TRT engine maximum batch size.
| `enable_trt_overlap` | Set to `true` to partition available requests into 2 'microbatches' that can be run concurrently to hide exposed CPU runtime |
| `exclude_input_in_output` | Set to `true` to only return completion tokens in a response. Set to `false` to return the prompt tokens concatenated with the generated tokens  |

#### Run the following command to prepare tensorrt_llm model config.pbtxt

```
python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:/tmp/llama/7B/trt_engines/fp16/4-gpu/,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600
```

*triton_model_repo/postprocessing/config.pbtxt*

Mandatory ones are
| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 64 |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `${HF_LLAMA_MODEL}`|
| `tokenizer_type` | The type of the tokenizer for the model, `t5`, `auto` and `llama` are supported. In this example, the type should be set to `llama` |
| `postprocessing_instance_count` | Here setting to 1 |

#### Run the following command to prepare postprocessing config.pbtxt
```
python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:llama,triton_max_batch_size:64,postprocessing_instance_count:1
```

*triton_model_repo/ensemble/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | Here setting to 64 |

#### Run the following command to prepare ensemble config.pbtxt
```
python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt triton_max_batch_size:64
```

### 7. Launch Triton server

Run this command to launch Triton server. Set `--world_size` = number of GPUs aka TP degree.

```
python3 scripts/launch_triton_server.py --world_size=4 --model_repo=/workspace/tensorrtllm_backend/triton_model_repo
```

### 8. Query the server with the Triton generate endpoint
[Query the server with the Triton generate endpoint](https://github.com/triton-inference-server/tensorrtllm_backend#query-the-server-with-the-triton-generate-endpoint)

```
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```
