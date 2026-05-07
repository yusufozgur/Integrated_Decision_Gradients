import torch
from typing import Optional, Union

from integrated_decision_gradients._utils import _scalar_output, _compute_delta, _compute_convergence_delta


def _idg_step(
    model: torch.nn.Module,
    baseline: torch.Tensor,
    delta: torch.Tensor,
    alphas: torch.Tensor,
    target: Optional[Union[int, torch.Tensor]],
    B: int,
    extra_dims: tuple,
    weight: float,
) -> torch.Tensor:
    """
    Compute the IDG weighted-gradient contribution for a chunk of alpha values.

    Implements line 12 of Algorithm 1:
        IDG[i] = (∂F/∂x) · (∂F/∂α) · weight
    where weight = 1 / (N * samples[i]).

    Returns a (B, ...) tensor accumulated over the chunk.
    """
    chunk = alphas.shape[0]
    alphas_v = alphas.view(chunk, 1, *extra_dims)           # (chunk, 1, ...)

    interpolated = (
        baseline.unsqueeze(0) + alphas_v * delta.unsqueeze(0)
    ).requires_grad_(True)                                  # (chunk, B, ...)

    flat = interpolated.view(chunk * B, *delta.shape[1:])
    output = _scalar_output(model, flat, target)            # (chunk*B,)
    output_cb = output.view(chunk, B)                       # (chunk, B)

    # ∂F/∂x at each interpolated point — shape (chunk, B, ...)
    grads, = torch.autograd.grad(
        outputs=output_cb.sum(),
        inputs=interpolated,
        create_graph=False,
        retain_graph=True,
    )

    # ∂F/∂α = ∇_x F · Δ  (chain rule: dx/dα = Δ along straight-line path)
    feat_dims = tuple(range(2, 2 + len(extra_dims)))
    logit_rate = (grads * delta.unsqueeze(0)).sum(dim=feat_dims)    # (chunk, B)
    logit_rate_v = logit_rate.view(chunk, B, *((1,) * len(extra_dims)))

    # Weighted integrand: (∂F/∂x_i) · (∂F/∂α) · weight
    weighted = grads * logit_rate_v * weight                # (chunk, B, ...)
    return weighted.detach().sum(dim=0)                     # (B, ...)


# ---------------------------------------------------------------------------
# Pre-characterisation pass (Algorithm 1, lines 3–7)
# ---------------------------------------------------------------------------

def _precharacterise(
    model: torch.nn.Module,
    baseline: torch.Tensor,
    delta: torch.Tensor,
    target: Optional[Union[int, torch.Tensor]],
    N: int,
    M: int,
    B: int,
) -> list:
    """
    Pilot pass: evaluate F at N+1 uniformly-spaced points and allocate M IDG
    sub-steps proportionally to the logit change in each of the N intervals.

    Algorithm 1, lines 3–7:
        samples[0]   = 0  (sentinel, unused)
        samples[i+1] = round( |ΔF_i| / |ΔF_total| * M )   for i = 0..N-1

    Any interval that rounds to 0 is given 1 step so no region is skipped.
    If the path is completely flat the budget is distributed uniformly.

    Returns a list of length N+1.
    """
    alphas_pilot = torch.linspace(0.0, 1.0, N + 1, device=baseline.device)
    shape_extra = (1,) * (delta.dim() - 1)

    with torch.no_grad():
        pts = (
            baseline.unsqueeze(0)
            + alphas_pilot.view(N + 1, 1, *shape_extra) * delta.unsqueeze(0)
        )                                                           # (N+1, B, ...)
        flat_pts = pts.view((N + 1) * B, *delta.shape[1:])
        f_vals = _scalar_output(model, flat_pts, target).view(N + 1, B)  # (N+1, B)

        # Mean absolute logit change per interval across batch
        dF = (f_vals[1:] - f_vals[:-1]).abs().mean(dim=1)          # (N,)
        total = dF.sum()

        if total < torch.finfo(dF.dtype).eps:
            # Flat path: distribute uniformly
            base = M // N
            remainder = M - base * N
            raw = [base + (1 if i < remainder else 0) for i in range(N)]
        else:
            proportions = dF / total                                # (N,)
            raw = [max(1, round((p * M).item())) for p in proportions]

    return [0] + raw   # samples[0] = 0 sentinel (Algorithm 1, line 3)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_integrated_decision_gradients(
    model: torch.nn.Module,
    input: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    target: Optional[Union[int, torch.Tensor]] = None,
    n_steps: int = 50,
    adaptive: bool = True,
    n_prechar_steps: Optional[int] = None,
    batch_size: int = 1,
    return_convergence_delta: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute Integrated Decision Gradients (IDG) for a model with respect to a
    given input and baseline.

    IDG improves on Integrated Gradients (IG) by addressing the saturation
    effect: gradients from regions where the output logit changes minimally
    contribute noise rather than signal. IDG weights each gradient by
    (∂F/∂α) — the rate of change of the target logit along the path —
    concentrating attribution mass in the "decision region" where the model
    transitions from uncertain to confident.

    When adaptive=True (default), Algorithm 1 from the paper is used: a cheap
    N-step pilot pass characterises the logit-α curve, then the M IDG steps
    are distributed non-uniformly so more samples fall in the decision region,
    further reducing Riemann-sum approximation error for the same step budget.

    When adaptive=False, n_steps uniform midpoint-rule samples are used
    (cheaper — no pilot pass — but less accurate near saturation boundaries).

    Reference: Walker et al., "Integrated Decision Gradients: Compute Your
    Attributions Where the Model Makes Its Decision", AAAI 2024.
    https://arxiv.org/abs/2305.20052

    Args:
        model:      The PyTorch model to explain. Must be callable and
                    differentiable.
        input:      Input tensor of shape (B, ...). requires_grad is set
                    automatically.
        baseline:   Baseline tensor (same shape as input). Represents the
                    "absence" of a feature — e.g. a zero tensor for images, or
                    a padding-token embedding for text. Defaults to zeros.
        target:     Index of the output neuron to attribute. For classification
                    this is typically the class index (int). Can also be a
                    tensor of shape (B,) for per-sample targets. If None, the
                    sum of all outputs is used.
        n_steps:    Total IDG integration steps M (higher → more accurate).
                    In adaptive mode this budget is distributed across the N
                    pilot intervals; in uniform mode it is the number of
                    evenly-spaced midpoint samples. Default 50.
        adaptive:   If True (default), use Algorithm 1 adaptive sampling.
                    A pilot pre-characterisation pass (N steps) allocates
                    more IDG steps to the decision region.
                    If False, use uniform midpoint-rule sampling — faster but
                    less accurate in the presence of saturation.
        n_prechar_steps:
                    Pilot steps N for the pre-characterisation pass
                    (adaptive=True only). The paper's ablation shows N=M is
                    optimal but doubles cost; we default to N = M // 2 as a
                    practical compromise. Ignored when adaptive=False.
        batch_size: Steps processed per forward/backward pass. Use 1 for
                    maximum memory efficiency; higher values trade memory for
                    speed. Defaults to 1.
        return_convergence_delta:
                    If True, also returns |Σ attributions − (F(x)−F(x'))|.
                    Values near 0 indicate a reliable approximation.

    Returns:
        attributions:       Tensor of shape matching `input`.
        convergence_delta:  (only if return_convergence_delta=True) Scalar.

    Example:
        >>> model = MyClassifier()
        >>> x = torch.rand(1, 3, 224, 224)
        >>> baseline = torch.zeros_like(x)

        >>> # adaptive (default) — best quality
        >>> attrs = integrated_decision_gradients(model, x, baseline, target=42)

        >>> # uniform — faster, no pilot pass
        >>> attrs = integrated_decision_gradients(
        ...     model, x, baseline, target=42, adaptive=False
        ... )
        >>> attrs.shape   # (1, 3, 224, 224)
    """
    if callable(getattr(model, "eval", None)):
        model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input)

    assert input.shape == baseline.shape, (
        f"input and baseline must have the same shape, "
        f"got {input.shape} vs {baseline.shape}"
    )

    B = input.shape[0]
    delta, extra_dims = _compute_delta(input, baseline)
    accumulated = torch.zeros_like(input)              # (B, ...)

    M = n_steps  # paper notation

    # ------------------------------------------------------------------
    # Branch 1: Adaptive sampling — Algorithm 1
    # ------------------------------------------------------------------
    if adaptive:
        N = n_prechar_steps if n_prechar_steps is not None else max(1, M // 2)

        # Lines 3–7: pilot pass → sub-step allocation per interval
        samples = _precharacterise(model, baseline, delta, target, N, M, B)
        # samples[0] = 0 (sentinel); samples[1..N] = sub-steps for interval i

        # Lines 8–14: main IDG integration with non-uniform alpha values
        for i in range(N):
            s_i = samples[i + 1]   # sub-steps allocated to interval i
            if s_i == 0:
                continue           # shouldn't happen (floor at 1), but safe

            # Alpha values within interval [i/N, (i+1)/N]  (Algorithm 1 line 10):
            #   α = i/N + j / (N * s_i)   for j = 0..s_i-1
            # We shift by 0.5 sub-steps (midpoint) to get O(h²) error within
            # each sub-interval — the paper uses j starting from 0, but the
            # midpoint variant is strictly more accurate for the same count.
            j_vals = torch.arange(s_i, device=input.device, dtype=input.dtype)
            alphas = i / N + (j_vals + 0.5) / (N * s_i)   # (s_i,)

            # Weight per sub-step = 1 / (N * s_i)  (line 12)
            weight = 1.0 / (N * s_i)

            for chunk_start in range(0, s_i, batch_size):
                chunk_alphas = alphas[chunk_start : chunk_start + batch_size]
                accumulated += _idg_step(
                    model, baseline, delta, chunk_alphas,
                    target, B, extra_dims, weight,
                )

    # ------------------------------------------------------------------
    # Branch 2: Uniform midpoint-rule sampling
    # ------------------------------------------------------------------
    else:
        # alpha_i = (i + 0.5) / M  for i = 0..M-1
        alphas = (torch.arange(M, device=input.device, dtype=input.dtype) + 0.5) / M
        weight = 1.0 / M

        for chunk_start in range(0, M, batch_size):
            chunk_alphas = alphas[chunk_start : chunk_start + batch_size]
            accumulated += _idg_step(
                model, baseline, delta, chunk_alphas,
                target, B, extra_dims, weight,
            )

    # ------------------------------------------------------------------
    # Scale by (x − x') then renormalise for completeness
    # ------------------------------------------------------------------
    raw_attributions = delta * accumulated              # (B, ...)

    with torch.no_grad():
        f_input    = _scalar_output(model, input,    target)   # (B,)
        f_baseline = _scalar_output(model, baseline, target)   # (B,)
        output_diff = f_input - f_baseline                     # (B,)

    # Rescale so attributions sum to F(x) − F(x') per sample
    eps     = torch.finfo(raw_attributions.dtype).eps
    raw_sum = raw_attributions.view(B, -1).sum(dim=-1)         # (B,)
    scale   = output_diff / (raw_sum + eps)                    # (B,)
    scale_v = scale.view(B, *((1,) * len(extra_dims)))         # (B, 1, ...)

    attributions = raw_attributions * scale_v                  # (B, ...)

    if not return_convergence_delta:
        return attributions

    convergence_delta = _compute_convergence_delta(attributions, output_diff)
    return attributions, convergence_delta
