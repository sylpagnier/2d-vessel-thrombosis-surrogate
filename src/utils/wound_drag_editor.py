"""Matplotlib wound region editor (sliders + live node highlight)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.widgets import Slider

from src.data_gen.lib.customer_geometry_import import apply_customer_mirrored_wound
from torch_geometric.data import Data


class WoundRegionEditor:
    """Provides sliders and a live scatter highlight to edit a wound patch."""

    def __init__(
        self,
        fig,
        ax_preview: Axes,
        ax_center: Axes,
        ax_width: Axes,
        data: Data,
        *,
        on_change: Callable[[Data], None],
        fill_color: str = "#a23a2e",
        wall_color: str = "#9a5a13",
    ) -> None:
        self.fig = fig
        self.ax_preview = ax_preview
        self.data = data
        self.on_change = on_change
        self.fill_color = fill_color
        self.wall_color = wall_color

        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        d_bar = float(data.d_bar.reshape(-1)[0].item()) if hasattr(data, "d_bar") else 1.0
        self.pos = pos * d_bar

        wall = getattr(data, "mask_wall", None)
        existing = getattr(data, "mask_wound", None)
        wall_b = wall.reshape(-1).bool().cpu().numpy() if wall is not None else np.zeros(pos.shape[0], dtype=bool)
        exist_b = existing.reshape(-1).bool().cpu().numpy() if existing is not None else np.zeros(pos.shape[0], dtype=bool)
        self.solid_mask = wall_b | exist_b

        init_center = getattr(data, "customer_wound_position_frac", 0.5)
        init_width = getattr(data, "customer_wound_width_frac", 0.15)

        self.slider_center = Slider(
            ax_center, "Wound Center", 0.0, 1.0, valinit=init_center, valstep=0.01
        )
        self.slider_width = Slider(
            ax_width, "Wound Width", 0.02, 0.80, valinit=init_width, valstep=0.01
        )
        
        # Style sliders
        for s in (self.slider_center, self.slider_width):
            s.ax.set_facecolor("#f1ece3")
            s.label.set_visible(False)
            s.valtext.set_color("#2b261f")
            s.valtext.set_fontsize(8.5)
            s.poly.set_color("#a23a2e")
            for spine in s.ax.spines.values():
                spine.set_color("#d9d0c2")
            s.on_changed(self._on_changed)

        self._scatter_wound = None
        self._scatter_healthy = None
        self._cids = []
        
        self.update_highlight()

    def _on_changed(self, _val: float) -> None:
        self.update_highlight()

    def update_highlight(self) -> None:
        if self._scatter_wound is not None:
            self._scatter_wound.remove()
        if self._scatter_healthy is not None:
            self._scatter_healthy.remove()

        c = self.slider_center.val
        w = self.slider_width.val
        
        # Apply wound locally to get the boolean masks
        new_data = apply_customer_mirrored_wound(
            self.data, enabled=True, position_frac=c, width_frac=w
        )
        wound_b = new_data.mask_wound.reshape(-1).bool().cpu().numpy()
        wall_b = new_data.mask_wall.reshape(-1).bool().cpu().numpy()

        if wall_b.any():
            self._scatter_healthy = self.ax_preview.scatter(
                self.pos[wall_b, 0], self.pos[wall_b, 1], c=self.wall_color, s=5.0, linewidths=0, zorder=4
            )
        if wound_b.any():
            self._scatter_wound = self.ax_preview.scatter(
                self.pos[wound_b, 0], self.pos[wound_b, 1], c=self.fill_color, s=12.0, linewidths=0, zorder=5
            )
            
        self.fig.canvas.draw_idle()
        self.on_change(new_data)

    def disconnect(self) -> None:
        for cid in self._cids:
            try:
                self.fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids = []
        if self._scatter_wound is not None:
            try:
                self._scatter_wound.remove()
            except Exception:
                pass
        if self._scatter_healthy is not None:
            try:
                self._scatter_healthy.remove()
            except Exception:
                pass
        self._scatter_wound = None
        self._scatter_healthy = None
