# keepbusy

A GPU idle monitor that keeps your GPUs busy with synthetic workloads after sustained idleness.

Useful for preventing GPU power-down or maintaining thermal stability in environments where idle GPUs are undesirable.

## Features

- Monitors GPU utilization via `nvidia-smi`
- Starts synthetic PyTorch workload only after sustained idleness (default: 10 minutes)
- Automatically yields when other GPU compute processes appear
- Supports multi-GPU systems with `DataParallel`
- Graceful shutdown via SIGINT/SIGTERM

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- NVIDIA drivers with `nvidia-smi` available

## Installation

```bash
pip install torch
```

## Usage

```bash
python keepbusy.py [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--util-threshold` | 5.0 | Treat a GPU as idle when utilization is below this percentage |
| `--idle-minutes` | 10.0 | Required continuous idle time before keepbusy starts |
| `--poll-interval` | 5.0 | Seconds between utilization checks |
| `--scale` | 2000 | Model scale factor for the synthetic workload |
| `--batch-size` | 6400 | Synthetic batch size used during keepbusy |
| `--disable-auto-interrupt` | False | Keep running even if another GPU compute process appears |

### Examples

Start with default settings (idle for 10 minutes triggers workload):
```bash
python keepbusy.py
```

Start after 5 minutes of idleness:
```bash
python keepbusy.py --idle-minutes 5
```

Run indefinitely, ignoring other processes:
```bash
python keepbusy.py --disable-auto-interrupt
```

## How It Works

1. **Monitor stage**: Polls GPU utilization at regular intervals
2. **Idle detection**: When all GPUs fall below the utilization threshold, a timer starts
3. **Keepbusy stage**: After sustained idleness, a synthetic neural network training loop runs
4. **Auto-yield**: If other GPU processes appear (2+ compute PIDs), returns to monitor stage

## Stopping

Press `Ctrl+C` to gracefully interrupt. The script handles SIGINT and SIGTERM signals.
