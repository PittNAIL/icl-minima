import torch
import torch.nn as nn

from dataclasses import asdict, dataclass

from tqdm import trange
from transformers import GPT2Config, GPT2Model


@dataclass(frozen=True)
class TransformerConfig:
    """A transformer configuration."""

    n_embd: int
    n_head: int
    n_layer: int

    attn_pdrop: int = 0.0
    embd_pdrop: int = 0.0
    resid_pdrop: int = 0.0
    use_cache: bool = False

    @property
    def dict(self) -> dict[str, int | bool]:
        """Returns a dictionary representation of a config."""

        return asdict(self)


CONFIGS: dict[str, TransformerConfig] = {
    "pico": TransformerConfig(n_embd=32, n_head=1, n_layer=1),
    "tiny": TransformerConfig(n_embd=64, n_head=2, n_layer=3),
    "small": TransformerConfig(n_embd=128, n_head=4, n_layer=6),
    "standard": TransformerConfig(n_embd=256, n_head=8, n_layer=12),
}


class GPT2(nn.Module):
    """GPT2 transformer model."""

    def __init__(
        self,
        config: TransformerConfig,
        n_dims: int,
        n_positions: int,
        out_dims: int,
    ) -> None:
        """Initializes the transformer model."""

        super().__init__()

        self._model = GPT2Model(GPT2Config(**config.dict, n_positions=n_positions))
        self._in = nn.Linear(n_dims, config.n_embd)
        self._out = nn.Linear(config.n_embd, out_dims)

        self.criterion = nn.MSELoss()

    def _interleave(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Interleaves the x's and the y's into a single sequence."""

        return torch.stack((xs, ys), dim=-1).view(xs.shape[0], -1, 1)

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        idxs = torch.arange(ys.shape[1])
        zs = self._interleave(xs, ys)

        embs = self._in(zs)
        out = self._model(inputs_embeds=embs).last_hidden_state
        preds = self._out(out)

        if ys.dim() == 2:
            return preds[:, ::2, 0][:, idxs]

        return preds[:, ::2, :][:, idxs]

    def reset_weights(self) -> None:
        """Resets all model weights."""

        @torch.no_grad()
        def weight_reset(module: nn.Module) -> None:
            # Checks if the current module has `reset_parameters`, and calls it if callable
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                module.reset_parameters()

        # Applies fn recursively to every submodule
        # See: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self._model.apply(fn=weight_reset)

    def run_train(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        n_epochs: int,
        learning_rate: float,
        desc: str = "",
    ) -> tuple[torch.Tensor, float]:
        """Trains the model."""

        self.reset_weights()
        self.train()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        losses = []
        for _ in trange(n_epochs, desc=desc):
            self.optimizer.zero_grad()
            loss = self.criterion(self(xs, ys), ys)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        losses = torch.tensor(losses)

        return losses, losses.mean()

    def mse_eval(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        """Evaluates the model via computing the Mean Squared Error (MSE)."""

        self.eval()
        with torch.no_grad():
            mse_eval = self.criterion(self(xs, ys), ys).mean().item()

        return mse_eval


class PyTorchBaselineModel(nn.Module):
    """A baseline model implemented in PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        in_features: int,
        out_features: int,
        learning_rate: float,
        epochs: int,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        return self.model(batch).squeeze()

    def run_train(self, xs: torch.Tensor, ys: torch.Tensor) -> tuple[list[float], float]:
        """Trains the model."""

        self.reset_weights()
        self.model.train()
        losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.criterion(self.forward(xs), ys)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        losses = torch.tensor(losses)

        return losses, losses.mean()

    def mse_eval(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        """Evaluates the model via computing the Mean Squared Error (MSE)."""

        self.eval()
        with torch.no_grad():
            mse_eval = self.criterion(self.forward(xs), ys).mean().item()

        return mse_eval

    def reset_weights(self) -> None:
        """Resets all model weights."""

        @torch.no_grad()
        def weight_reset(module: nn.Module) -> None:
            # Checks if the current module has `reset_parameters`, and calls it if callable
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                module.reset_parameters()

        # Applies fn recursively to every submodule
        # See: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.model.apply(fn=weight_reset)


class TwoLayerNeuralNetwork(PyTorchBaselineModel):
    """Two-Layer Neural Network (2NN)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float,
        epochs: int,
        hidden_features: int = 100,
    ) -> None:
        super().__init__(
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features),
            ),
            in_features,
            out_features,
            learning_rate,
            epochs,
        )


BASELINE_MODELS: dict[str, nn.Module] = {
    "2nn": TwoLayerNeuralNetwork,
}
