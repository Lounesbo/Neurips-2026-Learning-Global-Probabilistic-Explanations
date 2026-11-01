# -----------------------------------------------------------------------------
#
# Displayer: one figure or table per series
#
# Every figure draws the quantity the theory bounds next to its bound:
#   eps_sweep    gap err(K) - opt against epsilon, with y = epsilon: mean over
#                runs, and the range from the best to the worst run, since the
#                theorem bounds every run with probability 1 - delta
#   delta_sweep  failure frequency against delta, with y = delta (Wilson 95%)
#   d/k/c_sweep  queries against the swept value, with the closed form
#                (2d + 1) b n of IKM, and the IKM calls against N = 8dc/theta^2;
#                hollow markers flag the points whose block size was capped
#   trap         err against k for SE, GE and opt
#   datasets     err against k per dataset, and the LaTeX table of the paper
#                at a fixed k (--table, 5 by default), from the same runs
#
# Usage:  python displayer.py exps/eps_sweep/*.csv
#         python displayer.py exps/trap            (folder: its latest campaign)
#         python displayer.py --table 3 exps/datasets/*.csv
#         python displayer.py --no-tex ...   (when LaTeX or cm-super is missing)
#
# -----------------------------------------------------------------------------

import argparse
import glob
import re
import shutil
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Displayer:
    COLORS = {"SE": "#3E65FE", "GE": "#FE3E65", "opt": "#555555", "bound": "#48B52D"}
    MARKERS = {"SE": "o", "GE": "s"}

    def __init__(self, output_dir: str = "./exps", usetex: bool = True):
        self.output_dir = Path(output_dir)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
        if usetex and shutil.which("latex"):
            matplotlib.rcParams.update({"text.usetex": True,
                                        "text.latex.preamble": r"\usepackage{amsmath,amssymb}"})
        plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    # -- Loading ---------------------------------------------------------------

    @staticmethod
    def resolve(paths: List[str]) -> List[str]:
        """
        Expands wildcards here rather than in the shell, which PowerShell does
        not do. A folder stands for the latest campaign it holds: the files
        <series>_<tag>_shard<i>of<n>.csv sharing the most recent tag.
        """
        files = []
        for path in paths:
            if Path(path).is_dir():
                csvs = sorted(Path(path).glob("*.csv"))
                tags = {re.sub(r"_shard\d+of\d+$", "", p.stem): p for p in csvs}
                if not tags:
                    raise FileNotFoundError(f"No CSV in {path}")
                latest = max(tags)
                files += [str(p) for p in csvs if re.sub(r"_shard\d+of\d+$", "", p.stem) == latest]
            else:
                matches = sorted(glob.glob(path))
                if not matches:
                    raise FileNotFoundError(f"No file matches {path}")
                files += matches
        return files

    @staticmethod
    def load(paths: List[str]) -> pd.DataFrame:
        """
        Concatenates CSV files. Several campaigns may be given (e.g. the small
        and the medium datasets, or shards started separately): a run present
        in several campaigns is taken from the most recent one, so that a rerun
        replaces the older results.
        """
        files = Displayer.resolve(paths)
        frames = []
        for f in files:
            print(f"Reading {f}")
            frame = pd.read_csv(f)
            frame["campaign"] = re.sub(r"_shard\d+of\d+$", "", Path(f).stem)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        # A run present in several campaigns is taken from the most recent one.
        key = ["series", "task", "value", "run", "explainer"]
        df = df.sort_values("campaign", kind="stable")
        superseded = df.duplicated(subset=key, keep="last")
        for (task, campaign), n in df[superseded].groupby(["task", "campaign"]).size().items():
            print(f"Ignoring {n} rows of {task} from {campaign} (superseded)")
        return df[~superseded].reset_index(drop=True)

    def _save(self, name: str) -> None:
        for ext in ("pdf", "png"):
            plt.savefig(self.output_dir / "plots" / f"{name}.{ext}", bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved {name}")

    @staticmethod
    def _numeric(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    # -- Figures ---------------------------------------------------------------

    SIZE = (4.5, 3.2)  # every single-panel figure, so that they align side by side

    def eps_sweep(self, df: pd.DataFrame) -> None:
        """
        Two figures. eps_gap: the gap against epsilon, with the bound y = epsilon.
        eps_queries: the queries against epsilon, log-log, with the closed form
        (2d + 1) b n, which grows as epsilon^-4 ln(1/epsilon).
        """
        df = self._numeric(df)
        x = np.sort(df["value_num"].unique())

        plt.figure(figsize=self.SIZE)
        for name, g in df.groupby("explainer"):
            s = g.groupby("value_num")["gap"].agg(["mean", "min", "max"])
            plt.errorbar(s.index, s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]],
                         label=name, capsize=3, color=self.COLORS[name], marker=self.MARKERS[name])
        plt.plot(x, x, "--", color=self.COLORS["bound"], label=r"$\epsilon$")
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\mathsf{err}(K) - \mathrm{opt}$")
        plt.legend()
        self._save("eps_gap")

        plt.figure(figsize=self.SIZE)
        for name, g in df.groupby("explainer"):
            s = g.groupby("value_num")["queries"].mean()
            plt.plot(s.index, s.values, marker=self.MARKERS[name], color=self.COLORS[name], label=name)
        se = df[df["explainer"] == "SE"].groupby("value_num")["ikm_queries"].first()
        plt.plot(se.index, se.values, "--", color=self.COLORS["bound"], label=r"$(2d+1)\,bn$")
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(x, [f"{v:g}" for v in x])
        plt.minorticks_off()
        plt.xlabel(r"$\epsilon$")
        plt.ylabel("membership queries")
        plt.legend()
        self._save("eps_queries")

    def delta_sweep(self, df: pd.DataFrame) -> None:
        df = self._numeric(df[df["explainer"] == "SE"])
        s = df.groupby("value_num")["success"].agg(lambda v: (1 - v.astype(bool)).sum()).to_frame("fail")
        s["n"] = df.groupby("value_num").size()
        p, n, z = s["fail"] / s["n"], s["n"], 1.96
        center = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
        plt.figure(figsize=self.SIZE)
        plt.errorbar(s.index, p, yerr=[p - (center - half), (center + half) - p], marker="o",
                     capsize=3, color=self.COLORS["SE"], label="SE failure frequency")
        plt.plot(s.index, s.index, "--", color=self.COLORS["bound"], label=r"$\delta$")
        plt.xscale("log")
        plt.xlabel(r"$\delta$")
        plt.ylabel("failure frequency")
        plt.legend()
        self._save("delta_sweep")

    def cost_sweep(self, df: pd.DataFrame, name: str, xlabel: str, logx: bool = False) -> None:
        df = self._numeric(df)
        se = df[df["explainer"] == "SE"].groupby("value_num")
        capped = se["ikm_capped"].agg(lambda v: v.astype(bool).any())
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

        ax = axes[0]
        for expl, g in df.groupby("explainer"):
            s = g.groupby("value_num")["queries"].mean()
            ax.plot(s.index, s.values, color=self.COLORS[expl], label=expl)
            hollow = capped.reindex(s.index).fillna(False).astype(bool) & (expl == "SE")
            ax.scatter(s.index[~hollow], s.values[~hollow], marker=self.MARKERS[expl], color=self.COLORS[expl])
            ax.scatter(s.index[hollow], s.values[hollow], marker=self.MARKERS[expl],
                       facecolors="none", edgecolors=self.COLORS[expl], label="SE, capped" if hollow.any() else None)
        ax.plot(se["ikm_queries"].first().index, se["ikm_queries"].first().values, "--",
                color=self.COLORS["bound"], label=r"$(2d+1)\,bn$")
        ax.set_ylabel("membership queries")

        ax = axes[1]
        ax.plot(se["ikm_calls"].mean().index, se["ikm_calls"].mean().values, marker="o",
                color=self.COLORS["SE"], label="IKM estimates")
        ax.plot(se["ikm_max_calls"].first().index, se["ikm_max_calls"].first().values, "--",
                color=self.COLORS["bound"], label=r"$N = 8dc/\theta^2$")
        ax.set_ylabel("IKM estimates")

        for ax in axes:
            ax.set_yscale("log")
            if logx:
                ax.set_xscale("log", base=2)
            ax.set_xlabel(xlabel)
            ax.legend()
        self._save(name)

    def err_against_k(self, df: pd.DataFrame, name: str, by: str = "task") -> None:
        """One figure per value of `by`: per dataset, or one for the whole series
        when the task itself changes with k, as in the trap."""
        for task, t in df.groupby(by):
            t = self._numeric(t)
            plt.figure(figsize=self.SIZE)
            for expl, g in t.groupby("explainer"):
                s = g.groupby("value_num")["err"].agg(["mean", "min", "max"])
                plt.errorbar(s.index, s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]],
                             capsize=3, label=expl,
                             color=self.COLORS[expl], marker=self.MARKERS[expl])
            if t["opt"].notna().any():
                s = t.groupby("value_num")["opt"].first()
                plt.plot(s.index, s.values, ":", color=self.COLORS["opt"], label="opt")
            plt.xlabel(r"$k$")
            plt.ylabel(r"$\mathsf{err}(K)$")
            plt.xticks(sorted(t["value_num"].dropna().unique().astype(int)))
            if by == "task":  # one figure per dataset: the name is the only label
                plt.title(str(task))
            plt.legend()
            self._save(f"{name}_{str(task).replace(' ', '_')}" if by == "task" else name)

    # -- Table -----------------------------------------------------------------

    # Datasets of the table of the main text; all of them go to the appendix.
    MAIN = ["Breast Cancer (WI)", "Tic-Tac-Toe", "COMPAS", "Mushroom", "Credit-G"]

    def datasets_table(self, df: pd.DataFrame, k: int = 5) -> None:
        """
        Two tables: datasets_main_k{k}.tex (the datasets of MAIN) and
        datasets_all_k{k}.tex (all of them, for the appendix). Columns: err at
        a fixed k, and how often each explainer reaches opt over the whole
        k-sweep (|gap| < 1e-9). No run was capped, which the caption says.
        """
        if df[pd.to_numeric(df["k"]) == k].empty:
            raise ValueError(f"No run at k = {k}")
        if df["ikm_capped"].astype(bool).any():
            print("Warning: some runs were capped")
        rows = []
        order = df.groupby("task", sort=False)["d"].first().sort_values(kind="stable").index
        for task in order:
            every = df[df["task"] == task]
            t = every[pd.to_numeric(every["k"]) == k]
            se, ge = t[t["explainer"] == "SE"], t[t["explainer"] == "GE"]
            fmt = lambda v: f"${v.mean():.3f} \\pm {v.std():.3f}$"

            def hits(name: str) -> str:
                e = every[every["explainer"] == name]
                if e["gap"].isna().all():
                    return "--"
                return f"{int((e['gap'].abs() < 1e-9).sum())}/{len(e)}"

            rows.append({
                "Dataset": task, "$d$": int(t["d"].iloc[0]),
                "Acc.": f"{t['test_accuracy'].iloc[0]:.2f}",
                r"opt reached \textsc{se}": hits("SE"), r"opt reached \textsc{ge}": hits("GE"),
                r"\textsc{se}": fmt(se["err"]), r"\textsc{ge}": fmt(ge["err"]),
                "opt": f"{t['opt'].iloc[0]:.3f}" if t["opt"].notna().any() else "--",
                "Time (s)": f"{se['walltime'].mean():.0f}",
            })
        table = pd.DataFrame(rows)
        for name, part in (("all", table), ("main", table[table["Dataset"].isin(self.MAIN)])):
            if part.empty:
                continue
            path = self.output_dir / "tables" / f"datasets_{name}_k{k}.tex"
            part.to_latex(path, index=False, escape=False)
            print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures and tables of one series.")
    parser.add_argument("paths", nargs="+", help="CSV files of the series (all shards).")
    parser.add_argument("--table", type=int, default=5, help="k of the datasets table.")
    parser.add_argument("--no-tex", action="store_true", help="Matplotlib's own math rendering instead of LaTeX.")
    args = parser.parse_args()
    displayer = Displayer(usetex=not args.no_tex)
    df = Displayer.load(args.paths)
    series = df["series"].iloc[0]
    handlers = {
        "eps_sweep": lambda: displayer.eps_sweep(df),
        "delta_sweep": lambda: displayer.delta_sweep(df),
        "d_sweep": lambda: displayer.cost_sweep(df, "d_sweep", r"$d$", logx=True),
        "k_sweep": lambda: displayer.cost_sweep(df, "k_sweep", r"$k$"),
        "c_sweep": lambda: displayer.cost_sweep(df, "c_sweep", r"$c$"),
        "trap": lambda: displayer.err_against_k(df, "trap", by="series"),
        "datasets": lambda: (displayer.err_against_k(df, "datasets"),
                             displayer.datasets_table(df, args.table)),
    }
    handlers[series]()


if __name__ == "__main__":
    main()
