"""Optional Torch model boundary, imported only for Torch SHAP explainers."""
import torch


class TorchInputAdapter(torch.nn.Module):
    """Keep GradientExplainer's sampled tensors compatible with trained weights.

    SHAP can allocate samples using Torch's default dtype, even for a float64
    model. Convert at the model boundary; never mutate the original weights or
    Torch's process-wide default dtype.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        reference = next(model.parameters(), None)
        if reference is None:
            reference = next(model.buffers(), None)
        self.input_dtype = reference.dtype if reference is not None else torch.get_default_dtype()
        self.input_device = reference.device if reference is not None else None

    def forward(self, values):
        return self.model(values.to(dtype=self.input_dtype, device=self.input_device))
