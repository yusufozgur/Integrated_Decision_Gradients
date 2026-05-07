import torch
from typing import Optional, Union


def _scalar_output(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: Optional[Union[int, torch.Tensor]],
) -> torch.Tensor:
    """Return a (B,) scalar output for the chosen target.

    Handles models that return:
      - (B, C)  — standard multi-class logits
      - (B,)    — already a scalar per sample (regression, single-output)
      - (B, 1)  — single-output squeezable to (B,)
    """
    out = model(x)

    if out.dim() == 1:
        return out

    if out.dim() >= 2 and out.shape[-1] == 1:
        out = out.squeeze(-1)
        if out.dim() == 1:
            return out

    if target is None:
        return out.sum(dim=-1)
    if isinstance(target, int):
        return out[:, target]
    target_t = target.to(device=x.device)
    return out.gather(1, target_t.view(-1, 1)).squeeze(1)


def _compute_convergence_delta(
    attributions: torch.Tensor,
    output_diff: torch.Tensor,
) -> torch.Tensor:
    """Return sum(attributions) − sum(F(x) − F(x')).

    Values near 0 indicate the Riemann-sum approximation satisfies completeness.
    """
    return attributions.sum() - output_diff.sum()


def _compute_delta(
    input: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[torch.Tensor, tuple]:
    """Return (delta, extra_dims) for straight-line path integration.

    delta      = input - baseline, shape (B, ...)
    extra_dims = singleton dims for every non-batch axis, used for broadcasting
                 alpha scalars over the feature dimensions.
    """
    delta = input - baseline
    extra_dims = (1,) * (input.dim() - 1)
    return delta, extra_dims
