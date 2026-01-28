---
layout: post
title: "GPU Workload Distribution Patterns"
date: 2025-01-20
description: "From crypto mining to AI training: strategies for efficiently distributing ML workloads across a homelab GPU cluster."
---

# GPU Workload Distribution Patterns

I have a confession: my GPU cluster started as a crypto mining operation.

Back when Ethereum mining was profitable, I built a sizeable homelab filled with NVIDIA 3000 series GPUs. RTX 3080s, 3090s, cards stacked in custom rigs running 24/7. It was loud, it was hot, and it printed money. Then the Ethereum merge happened, proof-of-stake replaced proof-of-work, and suddenly I had a room full of expensive hardware with nothing to mine.

I switched some holdings to staking, but the GPUs? They needed a new purpose. That's when I pivoted to AI training, and I haven't looked back since.

## From Mining to Machine Learning

The transition wasn't as smooth as I expected. Mining is embarrassingly parallel: every GPU does the same thing independently. AI training is different. You need to coordinate GPUs, synchronize gradients, manage memory carefully, and handle the complexity of distributed systems.

My homelab now runs PyTorch, TensorFlow, and Axolotl for fine-tuning large language models. I've experimented with dozens of open-source models: Dolphin by Eric Hartford, GPT OSS 120B, Llama variants, Mistral, and many others. The 3000 series cards have 24GB of VRAM on the 3090s, which is enough to fine-tune 7B parameter models comfortably and run inference on much larger ones with quantization.

## Why I Train My Own Models

The most interesting project I've worked on: training an AI on my own code.

I fed years of my personal projects, coding style, comments, and commit messages into a fine-tuning pipeline. The goal was to create a model that understands how I think about code, my naming conventions, my architectural preferences, my typical patterns. The result is a coding assistant that feels like it actually knows me.

I've also been experimenting with AI agents that can play video games, learning from gameplay footage and reward signals. It's a fascinating application of reinforcement learning, and having local GPU power means I can iterate quickly without cloud costs eating into experiments.

## The Heterogeneity Problem

Even in my homelab, not every GPU is identical. I have a mix of 3080s and 3090s, different memory capacities, and varying thermal characteristics (some cards throttle more than others under sustained load). Your distribution strategy needs to handle this heterogeneity gracefully.

In production environments, this gets even messier:

- Mixed generations (V100s alongside A100s)
- Different memory capacities (16GB, 32GB, 80GB)
- Varying interconnects (PCIe vs NVLink)
- Spot/preemptible instances that disappear without warning

## Pattern 1: Data Parallelism

The simplest and most common pattern. Each GPU holds a complete copy of the model and processes different batches of data. Gradients are synchronized after each step.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size, model, dataset):
    setup(rank, world_size)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()  # Gradients automatically synchronized
        optimizer.step()
```

**When to use:** Models that fit in single GPU memory. Most computer vision and NLP fine-tuning tasks. This is my go-to for fine-tuning 7B models with Axolotl.

**Watch out for:** Communication overhead with large models. The all-reduce operation scales with model size, not data size.

## Pattern 2: Model Parallelism

When your model doesn't fit on a single GPU, you split the model itself across devices. Different layers live on different GPUs.

```python
class PipelinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # First half on GPU 0
        self.encoder = nn.Sequential(...).to('cuda:0')
        # Second half on GPU 1
        self.decoder = nn.Sequential(...).to('cuda:1')

    def forward(self, x):
        x = self.encoder(x.to('cuda:0'))
        x = self.decoder(x.to('cuda:1'))
        return x
```

**When to use:** Large models that exceed single GPU memory. I use this when running inference on 70B+ parameter models across multiple 3090s.

**Watch out for:** Pipeline bubbles. While GPU 1 processes batch N, GPU 0 sits idle unless you implement micro-batching.

## Pattern 3: Pipeline Parallelism with Micro-batching

The solution to pipeline bubbles: split each batch into micro-batches and keep all GPUs busy.

```python
# GPipe-style pipeline parallelism
def forward_with_microbatches(model_chunks, batch, num_microbatches):
    microbatches = batch.chunk(num_microbatches)
    outputs = []

    for mb in microbatches:
        x = mb
        for chunk in model_chunks:
            x = chunk(x)
        outputs.append(x)

    return torch.cat(outputs)
```

Libraries like DeepSpeed and FairScale implement this efficiently with automatic gradient checkpointing and memory optimization.

## Pattern 4: Tensor Parallelism

For massive matrix operations, split the tensors themselves across GPUs. A 4096x4096 weight matrix becomes four 4096x1024 shards.

```python
# Simplified tensor parallel linear layer
class TensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.shard_size = out_features // world_size
        self.weight = nn.Parameter(
            torch.randn(in_features, self.shard_size)
        )
        self.rank = rank

    def forward(self, x):
        local_output = x @ self.weight
        # All-gather to combine shards
        gathered = [torch.zeros_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered, local_output)
        return torch.cat(gathered, dim=-1)
```

**When to use:** Individual layers that are too large for single GPU memory. Attention layers in very large transformers.

## Handling Heterogeneous Hardware

Real clusters have mixed hardware. Here's how I handle it in my homelab:

### Dynamic Load Balancing

Assign work proportional to GPU capability:

```python
def get_gpu_weights(gpu_info):
    """Weight GPUs by their relative performance"""
    weights = []
    for gpu in gpu_info:
        if 'RTX 3090' in gpu.name:
            weights.append(1.0)
        elif 'RTX 3080' in gpu.name:
            weights.append(0.75)  # Less VRAM, slightly lower throughput
        else:
            weights.append(0.5)
    return normalize(weights)

# Distribute batch sizes proportionally
batch_sizes = [int(base_batch * w) for w in get_gpu_weights(gpus)]
```

### Memory-Aware Scheduling

Don't send a batch that won't fit:

```python
def schedule_batch(batch, available_gpus):
    batch_memory = estimate_memory(batch)

    for gpu in sorted(available_gpus, key=lambda g: g.free_memory, reverse=True):
        if gpu.free_memory > batch_memory * 1.2:  # 20% headroom
            return gpu

    # No GPU has enough memory, split the batch
    return split_and_reschedule(batch, available_gpus)
```

### Thermal Management

Mining taught me that sustained GPU loads generate serious heat. I monitor temperatures and throttle workloads before the hardware throttles itself:

```python
def check_thermal_headroom(gpu):
    if gpu.temperature > 80:
        return 0.5  # Reduce workload
    elif gpu.temperature > 70:
        return 0.8
    return 1.0

# Adjust batch sizes based on thermal state
for i, gpu in enumerate(gpus):
    batch_sizes[i] *= check_thermal_headroom(gpu)
```

## Monitoring and Profiling

You can't optimize what you can't measure. Essential metrics:

- **GPU utilization** - Should be >90% during compute
- **Memory utilization** - Track peak usage to right-size batches
- **Communication time** - all-reduce and all-gather overhead
- **Temperature** - Sustained training generates heat

```python
# Using PyTorch profiler
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

## Real-World Results

Fine-tuning a 7B parameter model on my homelab cluster (4x RTX 3090):

- **Naive data parallelism:** 52% GPU utilization, 6 hours for full fine-tune
- **With gradient accumulation:** 81% utilization, 3.5 hours
- **With optimized batch distribution:** 93% utilization, 2.8 hours

The hardware that used to mine Ethereum now trains models that help me write better code. I'd call that a successful pivot.

## Key Takeaways

1. **Repurpose what you have.** Mining rigs make surprisingly good ML clusters.
2. **Start simple.** Data parallelism with DDP handles most use cases.
3. **Profile before optimizing.** Find the actual bottleneck.
4. **Manage thermals.** Sustained AI training runs hotter than mining.
5. **Train on your own data.** A model fine-tuned on your code is incredibly useful.

---

*Running a homelab GPU cluster? Training your own models? I'd love to hear what you're building. Reach out anytime.*
