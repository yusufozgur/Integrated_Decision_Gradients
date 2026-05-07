import torch
from typing import Optional, Union

from integrated_decision_gradients._utils import _compute_delta, _scalar_output, _compute_convergence_delta


def get_integrated_gradients(
    model: torch.nn.Module,
    input: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    target: Optional[Union[int, torch.Tensor]] = None,
    n_steps: int = 50,
    batch_size: int = 1,
    return_convergence_delta: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute integrated gradients for a model with respect to a given input and baseline.

    Integrated gradients attribute a model's prediction to its input features by
    accumulating gradients along a straight-line path from the baseline to the input.
    See: https://arxiv.org/abs/1703.01365

    Args:
        model:      The PyTorch model to explain. Must be callable and differentiable.
        input:      Input tensor of shape (B, ...). Requires grad will be set
                    automatically.
        baseline:   Baseline tensor (same shape as input). Represents the "absence"
                    of a feature — e.g. a zero tensor for images, or a padding-token
                    embedding for text. Defaults to zeros if not provided.
        target:     Index of the output neuron to attribute. For classification this
                    is typically the class index (int). Can also be a tensor of
                    shape (B,) for per-sample targets. If None, the sum of all
                    outputs is used.
        n_steps:    Number of interpolation steps along the path (higher → more
                    accurate but slower). 20–300 is typical; default 50.
        batch_size: Number of interpolation steps to process at once. Use 1 for
                    maximum memory efficiency (large models/sequences), or higher
                    values to trade memory for speed. Defaults to 1.
        return_convergence_delta:
                    If True, also returns the convergence delta — the difference
                    between the sum of attributions and the model output difference
                    (f(x) − f(x')). Values near 0 indicate a reliable approximation.

    Returns:
        attributions:        Tensor of shape matching `input` containing the
                             integrated gradient attribution for each feature.
        convergence_delta:   (only if return_convergence_delta=True) Scalar tensor
                             measuring approximation error.

    Example:
        >>> model = MyClassifier()
        >>> x = torch.rand(1, 3, 224, 224)
        >>> baseline = torch.zeros_like(x)

        >>> # memory-efficient (large models/sequences)
        >>> attrs = integrated_gradients(model, x, baseline, target=42, batch_size=1)

        >>> # faster, uses more memory
        >>> attrs = integrated_gradients(model, x, baseline, target=42, batch_size=10)
        >>> attrs.shape   # (1, 3, 224, 224)
    """
    # if it has eval function, such as nn.Module, call it
    if callable(getattr(model, 'eval', None)):
        model.eval()

    # Default baseline: zero tensor
    if baseline is None:
        baseline = torch.zeros_like(input)

    assert input.shape == baseline.shape, (
        f"input and baseline must have the same shape, "
        f"got {input.shape} vs {baseline.shape}"
    )

    B = input.shape[0]
    delta, extra_dims = _compute_delta(input, baseline)
    accumulated_grads = torch.zeros_like(input)     # (B, ...)

    alphas = (torch.arange(n_steps, device=input.device) + 0.5) / n_steps

    # Process steps in chunks of `batch_size` to control peak memory usage.
    # Interpolated tensor shape: (chunk, B, ...)
    for chunk_start in range(0, n_steps, batch_size):
        chunk_alphas = alphas[chunk_start : chunk_start + batch_size]   # (chunk,)
        chunk = chunk_alphas.shape[0]

        # Reshape alphas to broadcast over (chunk, B, ...).
        # chunk_alphas: (chunk, 1, 1, ...) so it broadcasts with delta (B, ...)
        chunk_alphas_v = chunk_alphas.view(chunk, 1, *extra_dims)       # (chunk, 1, ...)

        # interpolated: (chunk, B, ...)
        interpolated = (
            baseline.unsqueeze(0) + chunk_alphas_v * delta.unsqueeze(0)
        ).requires_grad_(True)

        # Flatten the chunk and batch dims together for the forward pass: (chunk*B, ...)
        flat_interp = interpolated.view(chunk * B, *input.shape[1:])
        output = model(flat_interp)                 # (chunk*B, n_classes) or (chunk*B,)

        # Select the target scalar per sample
        if target is not None:
            if isinstance(target, int):
                # Same class for every sample
                output = output[:, target]          # (chunk*B,)
            else:
                # Per-sample target: shape (B,) → tile to (chunk*B,)
                target_t = target.to(device=input.device)           # (B,)
                tiled = target_t.repeat(chunk)                      # (chunk*B,)
                output = output.gather(1, tiled.view(-1, 1)).squeeze(1)  # (chunk*B,)
        else:
            output = output.sum(dim=-1)             # (chunk*B,)

        grads, = torch.autograd.grad(outputs=output.sum(), inputs=interpolated)
        # grads: (chunk, B, ...) — sum over the chunk axis to accumulate
        accumulated_grads += grads.detach().sum(dim=0)  # (B, ...)

    # Midpoint rule: divide by n_steps (all weights are 1/n_steps)
    avg_grads = accumulated_grads / n_steps         # (B, ...)

    # Scale by the input–baseline difference (Eq. 1 in the paper)
    attributions = delta * avg_grads                # (B, ...)

    if not return_convergence_delta:
        return attributions

    # --- Convergence delta (optional) ---
    with torch.no_grad():
        f_input    = _scalar_output(model, input,    target)   # (B,)
        f_baseline = _scalar_output(model, baseline, target)   # (B,)

    convergence_delta = _compute_convergence_delta(attributions, f_input - f_baseline)
    return attributions, convergence_delta
