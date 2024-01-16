## Using Triton TensorRT-LLM Backend through building the container

1. Get TRTLLM Backend repo
```
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.7.1
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive
```

2. Build the Triton TRT-LLM backend container (this will also install `tensorrt_llm` automatically) but building the container can take 25 minutes.

```
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
```

3. Launch the docker container
```
docker run --rm -it --net host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all  triton_trt_llm bash
```

4. Build engines
```
cd tensorrt_llm/examples/gpt

# Download weights from HuggingFace Transformers
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# Convert weights from HF Tranformers to FT format
python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16

# Build TensorRT engines
python3 build.py --model_dir=./c-model/gpt2/1-gpu/ \
                 --world_size=1 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --parallel_build \
                 --output_dir=engines/fp16/1-gpu

```

5. Modify the config files

6. Launch Triton server
```
python3 scripts/launch_triton_server.py --world_size=4 --model_repo=/tensorrtllm_backend/triton_model_repo

```