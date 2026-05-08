# Integrated Decision Gradients

A PyTorch implementation of **Integrated Decision Gradients (IDG)**, an attribution method that improves upon Integrated Gradients by concentrating attribution mass in the region where the model actually makes its decision.

> Walker et al., "Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision", AAAI 2024. [[paper]](https://arxiv.org/abs/2305.20052)

---

## Background: The Saturation Problem

Integrated Gradients (IG) attributes a model's prediction by integrating input gradients along a straight-line path from a baseline to the input:

```
IG_i(x) = (x_i − x'_i) × ∫₀¹ (∂F/∂x_i)(x' + α(x − x')) dα
```

The completeness property guarantees that attributions sum to `F(x) − F(x')`. In practice, the integral is approximated by sampling uniformly along `α ∈ [0, 1]`.

The problem is that most models make their decision early: the output logit shoots up near `α = 0` and then plateaus. Uniform sampling wastes most steps in the saturated, flat region where `∂F/∂α ≈ 0` — those gradients are noisy and contribute little to the sum. Achieving low approximation error therefore requires very high step counts.

---

## Integrated Decision Gradients

IDG introduces two improvements:

### 1. Importance Factor

Each gradient sample is weighted by `∂F/∂α` — the rate of change of the model output along the interpolation path. Samples in the flat/saturated region are down-weighted automatically because `∂F/∂α ≈ 0` there, while samples in the "decision region" (where the logit rises steeply) carry high weight.

```
IDG_i(x) = (x_i − x'_i) × ∫₀¹ (∂F/∂x_i) · (∂F/∂α) dα
```

`∂F/∂α` is computed efficiently via the chain rule along the straight-line path: `∂F/∂α = ∇_x F · (x − x')`.

### 2. Adaptive Sampling

A cheap pilot pre-characterization pass evaluates the model at `N` uniformly-spaced points to map out where the logit changes most. The `M` main integration steps are then allocated non-uniformly — more samples in the decision region, fewer in the saturated region — reducing Riemann-sum error for the same computational budget.

---

## Installation

**From PyPI:**

```bash
pip install integrated-decision-gradients
# or with uv:
uv add integrated-decision-gradients
```

**From source:**

```bash
git clone https://github.com/yusufozgur/integrated_decision_gradients
cd integrated-decision-gradients
uv sync                   # install all dependencies into .venv
uv add --editable .       # install the package itself in editable mode
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.3

---

## Quick Start

```python
import torch
from integrated_decision_gradients import get_integrated_decision_gradients

model = ...          # any differentiable torch.nn.Module
x = torch.rand(1, 3, 224, 224)
baseline = torch.zeros_like(x)

# Adaptive sampling (default) — best accuracy per step
attrs = get_integrated_decision_gradients(
    model, x, baseline, target=42
)

# Uniform sampling — faster, no pilot pass
attrs = get_integrated_decision_gradients(
    model, x, baseline, target=42, adaptive=False
)

# Check completeness: sum(attrs) should equal F(x) - F(baseline)
attrs, delta = get_integrated_decision_gradients(
    model, x, baseline, target=42, return_convergence_delta=True
)
print(f"Completeness error: {delta.item():.2e}")
```

---

## API Reference

### `get_integrated_decision_gradients`

```python
get_integrated_decision_gradients(
    model,
    input,
    baseline=None,
    target=None,
    n_steps=50,
    adaptive=True,
    n_prechar_steps=None,
    batch_size=1,
    return_convergence_delta=False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | — | Differentiable PyTorch model |
| `input` | `Tensor (B, ...)` | — | Input to explain |
| `baseline` | `Tensor (B, ...)` | zeros | Reference point representing feature absence |
| `target` | `int` or `Tensor (B,)` | `None` | Output neuron to attribute; sums all outputs if `None` |
| `n_steps` | `int` | `50` | Total IDG integration steps M |
| `adaptive` | `bool` | `True` | Use adaptive sampling (Algorithm 1); set `False` for uniform |
| `n_prechar_steps` | `int` | `n_steps // 2` | Pilot pass steps N (adaptive mode only) |
| `batch_size` | `int` | `1` | Steps per forward/backward pass; increase to trade memory for speed |
| `return_convergence_delta` | `bool` | `False` | Also return the completeness error scalar |

**Returns:** attribution tensor of the same shape as `input`, and optionally the convergence delta.

---

### `get_integrated_gradients`

Reference implementation of standard Integrated Gradients (Sundararajan et al., 2017) for comparison.

```python
get_integrated_gradients(
    model,
    input,
    baseline=None,
    target=None,
    n_steps=50,
    batch_size=1,
    return_convergence_delta=False,
)
```

Parameters are identical to `get_integrated_decision_gradients`, minus `adaptive` and `n_prechar_steps`.

---

## Performance

Benchmarked on a logistic curve model with a sharp decision boundary. **Delta** is the completeness error `|Σ attributions − (F(x) − F(x'))|` — lower is better.

| Method | Steps | Func Calls | Delta |
|---|---|---|---|
| Integrated Gradients (captum) | 30 | 30 | -1.20×10⁻³ |
| Integrated Gradients (uniform) | 30 | 30 | -3.35×10⁻⁴ |
| IDG (uniform) | 30 | 30 | -1.19×10⁻⁷ |
| **IDG (adaptive sampling)** | **20** | **30** | **0.00** |

IDG with adaptive sampling achieves exact completeness with fewer integration steps than standard IG, using the same number of function calls.

---

## Completeness

Attributions satisfy the completeness property:

```
Σᵢ IDG_i(x) = F(x) − F(x')
```

This is enforced by rescaling raw attributions so they sum exactly to the output difference. The convergence delta returned by `return_convergence_delta=True` measures how far the raw (pre-rescaling) sum is from this target — values close to 0 indicate a well-resolved integral.

---

## Citation

```bibtex
@inproceedings{walker2024idg,
  title     = {Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision},
  author    = {Walker, Chase and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024},
  url       = {https://arxiv.org/abs/2305.20052},
}
```
