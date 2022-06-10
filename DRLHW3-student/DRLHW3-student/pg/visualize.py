from typing import Dict, List, Tuple, Set, Any
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from ipywidgets import (Layout,
                        VBox,
                        Dropdown,
                        FloatSlider)


class Plotter():

    default_colors = [
        "#82c0cc",  # nameless
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#1f77b4",  # muted blue
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf"   # blue-teal
    ]

    def __init__(self, dataframe_dict: Dict[str, List[pd.DataFrame]]):
        self.dataframe_dict = dataframe_dict

        for exp_name, frames in dataframe_dict.items():
            frame_lengths = set()
            for frame in frames:
                frame_lengths.add(frame.size)
            if len(frame_lengths) > 1:
                raise ValueError((f"Frame lengths in experiment {exp_name} are not the same."
                                  f" Unique frame lengths: {frame_lengths}. Make sure "
                                  f"you fix n-iterations and write-period within an experiment."))
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",)

        self.x_name = None
        self.y_name = None
        self.quantile_value = 0.25
        self.set_figure()
        self.set_components()
        self.fill_dropdowns()

    def fill_dropdowns(self) -> None:
        names, intersect_monotonic_names = self._all_column_names(self.dataframe_dict)
        self.select_yaxis.options = names
        self.select_xaxis.options = intersect_monotonic_names

    def set_figure(self) -> None:
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True)

    @staticmethod
    def _get_column_names(dataframes: List[pd.DataFrame]) -> Tuple[Set, Set]:
        names = tuple(dataframes[0].columns)
        for df in dataframes:
            if sorted(tuple(df.columns)) != sorted(names):
                raise ValueError("Column names do not match")

        intersect_monotonic_names = set(names)
        for df in dataframes:
            monotonic_names = []
            for name in names:
                diff = np.diff(df[name].to_numpy())
                diff = diff[~np.isnan(diff)]
                if np.all(diff >= 0):
                    monotonic_names.append(name)
            monotonic_names = set(monotonic_names)
            intersect_monotonic_names = intersect_monotonic_names & monotonic_names

        return set(names), intersect_monotonic_names

    @staticmethod
    def _all_column_names(dataframe_dict: Dict[str, List[pd.DataFrame]]) -> Tuple[Set, Set]:
        if len(dataframe_dict) == 0:
            return [], []

        intersect_ynames, intersect_xnames = None, None
        for frames in dataframe_dict.values():
            ynames, xnames = Plotter._get_column_names(frames)
            if intersect_xnames is None:
                intersect_xnames = xnames
                intersect_ynames = ynames
            else:
                intersect_xnames = intersect_xnames & xnames
                intersect_ynames = intersect_ynames & ynames

        return intersect_ynames, intersect_xnames

    def set_components(self) -> None:
        self.select_yaxis = Dropdown(
            options=[],
            value=None,
            description="Y axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_yaxis.observe(self.set_y_axis)

        self.select_xaxis = Dropdown(
            options=[],
            value=None,
            description="X axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_xaxis.observe(self.set_x_axis)

        self.quantile_slider = FloatSlider(
            value=self.quantile_value,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Quantile",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        )
        self.quantile_slider.observe(self.set_quantile)

    def set_quantile(self, change: Any) -> None:
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.quantile_value = change["new"]
        self.render_figure()

    def set_x_axis(self, change: Any) -> None:
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.x_name = change["new"]
        self.render_figure()

    def set_y_axis(self, change: Any) -> None:
        if (change["new"] == change["old"]) or change["name"] != "value":
            return None
        self.y_name = change["new"]
        self.render_figure()

    def _add_traces(self, dataframe: pd.DataFrame, color: str, legend_name: str) -> None:
        y_values = np.stack([df[self.y_name].to_numpy()
                             for df in dataframe], axis=0)

        median = np.quantile(y_values, 0.5, interpolation="nearest", axis=0)
        upper_quantile = np.quantile(
            y_values, 1.0 - self.quantile_value, interpolation="nearest", axis=0)
        lower_quantile = np.quantile(
            y_values, self.quantile_value, interpolation="nearest", axis=0)

        self.fig.add_trace(
            go.Scatter(
                x=dataframe[0][self.x_name],
                y=upper_quantile,
                mode="lines",
                legendgroup=legend_name,
                showlegend=False,
                line=dict(
                    color=color,
                    width=1,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x=dataframe[0][self.x_name],
                y=lower_quantile,
                mode="lines",
                fill="tonexty",
                legendgroup=legend_name,
                showlegend=False,
                line=dict(
                    color=color,
                    width=1,
                    shape="spline",
                    smoothing=0.7)
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x=dataframe[0][self.x_name],
                y=median,
                mode="lines",
                legendgroup=legend_name,
                name=legend_name,
                line=dict(
                    color=color,
                    width=2,
                    shape="spline",
                    smoothing=0.7)
            )
        )

    def render_figure(self) -> None:
        if self.x_name is None or self.y_name is None:
            return
        self.fig.data = []
        for index, (name, frames) in enumerate(self.dataframe_dict.items()):
            self._add_traces(frames, self.default_colors[index], name)

        self.fig.update_layout(
            yaxis={
                "title": self.y_name,
                "gridcolor": "gray",
            },
            xaxis={
                "title": self.x_name,
                "gridcolor": "gray",
            }
        )

    def __call__(self) -> VBox:
        out_display = VBox([
            self.select_xaxis,
            self.select_yaxis,
            self.quantile_slider,
            self.fig,
        ],
            layout=Layout(
                width="96%",
                height="660px",
                display="flex",
                flex_flow="column",
                justify_content="space-around",
                border="solid 2px gray"
        ))
        return out_display