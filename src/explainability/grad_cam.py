"""Gradient-weighted class activation mapping for the project backbones.

The implementation is deliberately dependency-light: it uses PyTorch hooks
directly instead of requiring a second explainability package.  The returned
map explains one class logit, not the post-softmax probability.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Generate Grad-CAM maps from a named convolutional feature block.

    A single instance may be reused for several images.  Call :meth:`close`
    when it is no longer needed, or use the instance as a context manager.

    Args:
        model: Classifier that returns logits shaped ``(batch, classes)``.
        target_layer: Feature block whose output has shape
            ``(batch, channels, height, width)``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(
        self,
        _module: nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Keep the feature tensor and attach its gradient hook."""
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError(
                "Grad-CAM target layer must return a 4-D tensor shaped "
                "(batch, channels, height, width)"
            )
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        """Store the gradient of the selected logit with respect to features."""
        self.gradients = gradient

    def __call__(
        self,
        inputs: torch.Tensor,
        target_class: Union[int, torch.Tensor] = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a normalized Grad-CAM map for each input image.

        Args:
            inputs: Normalized model input shaped ``(B, C, H, W)``.
            target_class: Either one class index for the full batch or a
                length-``B`` tensor of class indices.

        Returns:
            A tuple ``(maps, logits)``. ``maps`` has shape ``(B, H, W)`` and
            values in ``[0, 1]``; ``logits`` is detached from the graph.

        Raises:
            ValueError: If the input or target-class shape is invalid.
            RuntimeError: If the selected layer did not expose activations or
                gradients during the forward/backward pass.
        """
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch, channels, height, width)")

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(inputs)
        if logits.ndim != 2:
            raise ValueError("model must return logits shaped (batch, classes)")

        batch_size = logits.shape[0]
        if isinstance(target_class, int):
            targets = torch.full(
                (batch_size,),
                target_class,
                dtype=torch.long,
                device=logits.device,
            )
        else:
            targets = target_class.to(device=logits.device, dtype=torch.long)
            if targets.shape != (batch_size,):
                raise ValueError("target_class tensor must have shape (batch,)")

        selected_logits = logits.gather(1, targets[:, None]).sum()
        selected_logits.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "No Grad-CAM tensors were captured; verify the target layer "
                "participates in the selected class logit"
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        maps = F.interpolate(
            maps,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        flat = maps.flatten(1)
        minimum = flat.min(dim=1).values[:, None, None]
        maximum = flat.max(dim=1).values[:, None, None]
        maps = (maps - minimum) / (maximum - minimum).clamp_min(1e-12)

        if was_training:
            self.model.train()
        return maps.detach(), logits.detach()

    def close(self) -> None:
        """Remove the forward hook registered on the target layer."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "GradCAM":
        """Return this instance for use in a ``with`` block."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Release the registered hook when leaving a ``with`` block."""
        self.close()


def resolve_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return the last spatial feature block for a supported architecture.

    Explicit architecture-to-layer mapping is safer than walking the module
    tree and taking the last ``Conv2d``: the latter can silently select a layer
    inside a residual/dense block whose output is not what the classifier sees.

    Args:
        model: Instantiated project classifier.
        architecture: ``"resnet18"``, ``"densenet121"`` or ``"simple_cnn"``.

    Returns:
        The feature block used as the Grad-CAM target.

    Raises:
        ValueError: If the architecture is unsupported.
    """
    if architecture == "resnet18":
        return model.layer4[-1]
    if architecture == "densenet121":
        return model.features.denseblock4
    if architecture in {"simple_cnn", "small_cnn"}:
        return model.features[-1]
    raise ValueError(
        f"Unsupported Grad-CAM architecture: {architecture!r}. "
        "Choose resnet18, densenet121 or simple_cnn."
    )
