"""Matplotlib visualizations for odometry evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter


class EvalVisualizer:
    """Create trajectory, scatter/histogram, and summary plots."""

    def __init__(self, save_path: str | Path | Mapping[str, str | Path], dpi: int = 300) -> None:
        if isinstance(save_path, Mapping):
            save_path = save_path.get("traj", ".")
        self.save_path = Path(save_path).expanduser()
        self.dpi = dpi

    @staticmethod
    def gaussianfit(points_xy: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted Gaussian fit returning ``(mean, covariance)``."""

        points = np.asarray(points_xy, dtype=float).T
        weights = np.asarray(weights, dtype=float).reshape(-1)
        mean = np.average(points, axis=1, weights=weights)
        centered_points = points - mean[:, np.newaxis]
        covariance = np.cov(centered_points, aweights=weights)
        return mean, covariance

    def get_my_ellipse(
        self,
        t_err_list: np.ndarray,
        r_err_list: np.ndarray,
        confidence: float,
        linewidth: float | None = None,
        linecolor: str = "b",
    ) -> tuple[Ellipse, float, float]:
        """Build a confidence ellipse for translation/rotation errors."""

        t_err = np.asarray(t_err_list, dtype=float).reshape(-1, 1)
        r_err = np.asarray(r_err_list, dtype=float).reshape(-1, 1)
        mirrored = np.vstack(
            (
                np.hstack((t_err, r_err)),
                np.hstack((-t_err, r_err)),
                np.hstack((-t_err, -r_err)),
                np.hstack((t_err, -r_err)),
            )
        )
        mean, covariance = self.gaussianfit(mirrored, np.ones(mirrored.shape[0]))
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        std_deviation = np.sqrt(-2.0 * np.log(1.0 - confidence))
        semi_major_axis = float(std_deviation * np.sqrt(eigenvalues[0]))
        semi_minor_axis = float(std_deviation * np.sqrt(eigenvalues[1]))
        angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
        ellipse = Ellipse(
            mean,
            width=2.0 * semi_minor_axis,
            height=2.0 * semi_major_axis,
            angle=angle,
            edgecolor=linecolor,
            facecolor="none",
            linewidth=linewidth,
            linestyle="dashed",
        )
        return ellipse, semi_major_axis, semi_minor_axis

    def plot_traj(
        self,
        evaluator,
        sequence: str,
        plot_keys: list[str] | None = None,
        ignore_keys: set[str] | None = None,
        plot_show: bool = False,
        plot_save: bool = True,
    ) -> Path | None:
        """Plot GT and selected baseline trajectories."""

        plot_keys = plot_keys or evaluator.baselines_name
        font = fm.FontProperties(family="DejaVu Serif", size=14)
        font_legend = fm.FontProperties(family="DejaVu Serif", size=9)
        fig, ax = plt.subplots(figsize=(6, 6))

        for item in evaluator.gt_data:
            poses = np.asarray(list(next(iter(item.values())).values()))
            key = str(next(iter(item.keys())))
            ax.plot(poses[0, 0, -1], poses[0, 1, -1], linewidth=3, color="blue", marker="^", markersize=10)
            ax.plot(poses[:, 0, -1], poses[:, 1, -1], label=key, linewidth=3, color="black")
            ax.plot(poses[-1, 0, -1], poses[-1, 1, -1], marker="*", markersize=10, color="black")

        for item in evaluator.baselines_data:
            poses = np.asarray(list(next(iter(item.values())).values()))
            key = str(next(iter(item.keys())))
            if ignore_keys and key in ignore_keys:
                continue
            if key not in plot_keys:
                continue
            if key.lower() in {"model_a", "model-a", "model a"}:
                ax.plot(poses[:, 0, -1], poses[:, 1, -1], label=key, linestyle="-", linewidth=3, color="red")
                ax.plot(poses[-1, 0, -1], poses[-1, 1, -1], marker="*", markersize=10, color="red")
            else:
                ax.plot(poses[:, 0, -1], poses[:, 1, -1], label=key, linestyle="--", linewidth=2.5)
                ax.plot(poses[-1, 0, -1], poses[-1, 1, -1], marker="*", markersize=10, color=ax.lines[-1].get_color())

        legend = ax.legend(prop=font_legend, loc="best", ncol=2)
        legend.get_frame().set_alpha(0.5)
        ax.set_xlabel("X (m)", fontproperties=font)
        ax.set_ylabel("Y (m)", fontproperties=font)
        ax.grid(linestyle="dashed")
        output = self.save_path / f"{sequence}_traj.png"
        if plot_save:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.0, dpi=self.dpi)
        if plot_show:
            plt.show()
        plt.close(fig)
        return output if plot_save else None

    def plot_scatter_and_hist(
        self,
        steperrors_t: Mapping[str, list[float]],
        steperrors_r: Mapping[str, list[float]],
        plot_keys: list[str],
        sequence: str,
        plot_show: bool = False,
        plot_save: bool = True,
        bins_num: int = 30,
    ) -> list[Path]:
        """Plot relative translation/rotation errors with marginal histograms."""

        class HandlerEllipse(HandlerPatch):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                patch = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
                self.update_prop(patch, orig_handle, legend)
                patch.set_transform(trans)
                return [patch]

        outputs: list[Path] = []
        for plot_key in plot_keys:
            t_err = np.asarray(steperrors_t[plot_key], dtype=float).reshape(-1, 1)
            r_err = np.asarray(steperrors_r[plot_key], dtype=float).reshape(-1, 1)
            if t_err.size == 0 or r_err.size == 0:
                continue
            fig = plt.figure(figsize=(8, 8))
            gs = gridspec.GridSpec(4, 4)
            distances = np.power(200 * r_err + t_err, 0.3)
            ax_main = plt.subplot(gs[1:4, 0:3])
            ax_main.scatter(r_err, t_err, c=distances, cmap="plasma", alpha=0.8, marker=".")
            ax_main.set_xlabel("Relative Rotation Error (rad)")
            ax_main.set_ylabel("Relative Translation Error (m)")
            ellipses = [
                self.get_my_ellipse(t_err, r_err, 0.90, 2.5, "darkblue"),
                self.get_my_ellipse(t_err, r_err, 0.95, 2.5, "royalblue"),
                self.get_my_ellipse(t_err, r_err, 0.99, 2.5, "cornflowerblue"),
            ]
            for ellipse, semi_major, semi_minor in ellipses:
                ax_main.add_patch(ellipse)
                ax_main.axvline(x=semi_minor, linestyle="dashed", color="gray", linewidth=0.8)
                ax_main.axhline(y=semi_major, linestyle="dashed", color="gray", linewidth=0.8)
            ax_main.legend(
                handles=[item[0] for item in ellipses],
                labels=["90% confidence", "95% confidence", "99% confidence"],
                loc="upper right",
                handler_map={Ellipse: HandlerEllipse()},
            )
            x_max = max(2.2 * ellipses[-1][2], float(np.max(r_err)) * 1.05, 1e-9)
            y_max = max(2.2 * ellipses[-1][1], float(np.max(t_err)) * 1.05, 1e-9)
            ax_main.set_xlim(0, x_max)
            ax_main.set_ylim(0, y_max)

            ax_x = plt.subplot(gs[0, 0:3], sharex=ax_main)
            counts, bins = np.histogram(r_err, bins=bins_num, range=(0, x_max))
            ax_x.bar(0.5 * (bins[:-1] + bins[1:]), counts, width=np.diff(bins), color="cornflowerblue")
            ax_x.set_ylabel("Count")
            ax_x.set_title(plot_key)

            ax_y = plt.subplot(gs[1:4, 3], sharey=ax_main)
            counts, bins = np.histogram(t_err, bins=bins_num, range=(0, y_max))
            ax_y.barh(0.5 * (bins[:-1] + bins[1:]), counts, height=np.diff(bins), color="lightcoral")
            ax_y.set_xlabel("Count")
            plt.setp(ax_x.get_xticklabels(), visible=False)
            plt.setp(ax_y.get_yticklabels(), visible=False)
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            output = self.save_path / f"{sequence}_{plot_key}_scatter_and_hist.png"
            if plot_save:
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.01, dpi=self.dpi)
                outputs.append(output)
            if plot_show:
                plt.show()
            plt.close(fig)
        return outputs

    def plot_bar_and_box(self, steperrors_t, steperrors_r, result_ave_t, result_ave_r, sequence, plot_show=False, plot_save=True):
        """Plot aggregate bars and step-error boxplots."""

        keys = list(result_ave_t.keys())
        if not keys:
            return None
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        for ax, step_data, avg_data, ylabel, title in (
            (axs[0], steperrors_t, result_ave_t, "Relative Translation Error (m)", "Translation"),
            (axs[1], steperrors_r, result_ave_r, "Relative Rotation Error (rad)", "Rotation"),
        ):
            values = [step_data[key] for key in keys]
            ax.boxplot(values, showfliers=False, patch_artist=True)
            ax.set_xticks(range(1, len(keys) + 1))
            ax.set_xticklabels(keys, rotation=20, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, linestyle="dashed", alpha=0.5)
            ax2 = ax.twinx()
            ax2.bar(range(1, len(keys) + 1), [avg_data[key] for key in keys], alpha=0.25, color="steelblue")
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        output = self.save_path / sequence / f"{sequence}_bar_and_box.pdf"
        if plot_save:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.0)
        if plot_show:
            plt.show()
        plt.close(fig)
        return output if plot_save else None


evalVis = EvalVisualizer
