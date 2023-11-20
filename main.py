#!/usr/bin/env python
import argparse
import json

import torch

import matplotlib.pyplot as plt

from transformers import set_seed

from icl.function import NMinimaFunction
from icl.model import BASELINE_MODELS, CONFIGS, GPT2, PyTorchBaselineModel


plt.style.use("tableau-colorblind10")
set_seed(1_337)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("In-Context Learning Experiments")
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--model_config", type=str, help="model type", required=True)
    parser.add_argument("--n_dims", type=int, help="number of dimensions", default=1)
    parser.add_argument("--n_epochs", type=int, help="number of iterations", default=1_000)
    parser.add_argument("--n_minima", type=int, nargs="+", help="minima", default=[1, 2, 3, 4])
    parser.add_argument("--n_positions", type=int, help="maximum sequence length", default=1_024)
    parser.add_argument("--n_prompts", nargs="+", help="prompt counts", default=[8, 16, 32])

    return parser.parse_args()


def main() -> None:
    """Trains the model on In-Context Learning (ICL) tasks."""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert args.n_dims == 1
    f = NMinimaFunction()
    n_points = args.n_positions // (2 * args.n_dims)

    model = GPT2(CONFIGS[args.model_config], args.n_dims, args.n_positions, out_dims=args.n_dims)
    model.to(device)

    nrows, ncols = len(args.n_minima), len(args.n_prompts)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 16),
        dpi=256,
        gridspec_kw={"hspace": 0.5, "wspace": 0.5},
    )

    row, col = 0, 0
    mse_stats = {}
    for n_minima in args.n_minima:
        # Garg, et al. 2023 did not do epochs over the dataset, and instead used iterations as they
        # randomly sampled from the dataset. This direction is interesting, but most of the time, we
        # think of training in terms of a fixed dataset and epochs over that dataset. Note that in
        # this case we can just use `n_prompts` as the batch size since it is one of 8, 16, 32.
        #
        # While very large generative LLMs usually do not suffer from overfitting as they are not
        # trained on the entire dataset, and modeling that behavior is interesting, it is also
        # important to consider the alternative (epoch-based) training.
        #
        # NOTE: The following for `n_positions` works out nicely as we have `n_dims` operands
        # (x_1, ..., x_n) and the result y.
        #
        # Also, note that we have the same input for all functions, just that the functions change.
        mse_stats[f"{n_minima=}"] = {}

        # Generate data, select minima, and shuffle
        size = 512 * max(args.n_prompts) * n_points * args.n_dims
        xs = torch.randn(size)

        idxs = torch.randperm(len(xs))[:n_minima]
        minima = torch.stack((xs[idxs], torch.randn(n_minima)), dim=-1)

        ys = f(minima, xs)
        zs = torch.stack((xs, ys), dim=-1)
        zs = zs[torch.randperm(zs.size()[0])]
        xs, ys = zs.T.to(device)

        for n_prompts in args.n_prompts:
            mse_stats[f"{n_minima=}"][f"{n_prompts=}"] = {}
            ax = axs[row, col]

            chunk = n_prompts * n_points * args.n_dims
            r = lambda x: x.view(n_prompts, n_points, args.n_dims)

            xs_train, xs_eval, ys_train, ys_eval = (
                r(xs[:chunk]),
                r(xs[-chunk:]),
                r(ys[:chunk]),
                r(ys[-chunk:]),
            )

            # Label format for the legend
            label = lambda n, t: f"{n.ljust(len(args.model_config))} {t:.4f}"

            # Transformer model
            losses_train, mse_train = model.run_train(
                xs_train,
                ys_train,
                args.n_epochs,
                args.learning_rate,
                f"Prompts={n_prompts:<2} | Minima={n_minima:<2}",
            )
            mse_eval = model.mse_eval(xs_eval, ys_eval)
            ax.plot(losses_train, label=label(args.model_config, mse_train), linewidth=2.0)
            mse_stats[f"{n_minima=}"][f"{n_prompts=}"][args.model_config] = mse_eval

            # Baseline models
            for bm_name, _bm in BASELINE_MODELS.items():
                bm = _bm(args.n_dims, 1, args.learning_rate, args.n_epochs)
                bm.to(device)
                losses_bm_train, mse_bm_train = bm.run_train(xs_train, ys_train.squeeze())
                ax.plot(losses_bm_train, label=label(bm_name, mse_bm_train), linewidth=2.0)
                mse_bm_eval = bm.mse_eval(xs_eval, ys_eval.squeeze())
                mse_stats[f"{n_minima=}"][f"{n_prompts=}"][bm_name] = mse_bm_eval

            # Plotting logic
            ax.set_title(
                f"{n_minima} Minima | {n_prompts} Prompts",
                fontname="monospace",
                fontsize=14,
            )
            ax.legend(
                bbox_to_anchor=(1.28, 1.0),
                loc="upper right",
                prop={"family": "monospace", "size": 12},
            )
            ax.grid(linestyle="dashed")
            ax.tick_params(axis="both", which="major", labelsize=12)

            if row == nrows - 1:
                ax.set_xlabel("Epochs", fontsize=12)

            if col == 0:
                ax.set_ylabel("Mean Squared Error", fontsize=12)

            col += 1
            if col == ncols:
                row += 1
                col = 0

            if row == nrows:
                break

    fig.savefig(f"{args.model_config}_loss.png", bbox_inches="tight")

    with open(f"{args.model_config}_mse.json", "w", encoding="utf-8") as file:
        json.dump(mse_stats, file, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
