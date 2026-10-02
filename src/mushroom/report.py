"""Self-contained static report (``site/index.html``) with interactive charts."""

from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.offline

from mushroom.config import (
    AMBER,
    CLASS_COLORS,
    PROCESSED_CSV,
    RESULTS_JSON,
    SITE_HTML,
    SLATE,
    TABLES_DIR,
    TEAL,
)
from mushroom.data import load_processed
from mushroom.figures import MODEL_COLORS
from mushroom.mca import MCA

REPO = "https://github.com/Pchambet/mushroom-project"
GRIDLINE = "rgba(100,116,139,0.22)"


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"{name}.csv")


def _layout(fig: go.Figure, **kwargs: object) -> go.Figure:
    defaults: dict[str, object] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "color": SLATE, "size": 13},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
        "height": 400,
        "legend": {"orientation": "h", "y": -0.2},
    }
    fig.update_layout(defaults | kwargs)
    fig.update_xaxes(gridcolor=GRIDLINE, zeroline=False)
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False)
    return fig


def _div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def axis_curve_chart(results: dict) -> str:
    curve = _table("axis_curve")
    fig = go.Figure()
    for model, color in MODEL_COLORS.items():
        sub = curve[curve["model"] == model]
        fig.add_scatter(
            x=sub["n_axes"],
            y=100 * sub["mean"],
            error_y={"array": 100 * sub["std"], "thickness": 1},
            name=model,
            mode="lines+markers",
            line={"color": color, "width": 2.5},
            hovertemplate="%{x} axes: %{y:.2f}%<extra>" + model + "</extra>",
        )
    fig.add_hline(
        y=100 * results["odour_accuracy"],
        line={"dash": "dash", "color": SLATE},
        annotation_text="odour alone",
    )
    return _div(
        _layout(
            fig,
            xaxis={
                "type": "log",
                "title": "MCA axes kept (log scale)",
                "tickvals": [1, 2, 3, 5, 10, 20, 40, results["n_axes"]],
                "range": [-0.05, 2.05],
            },
            yaxis={"title": "Out-of-fold accuracy (%)", "range": [80, 100.5]},
        )
    )


def eta_chart() -> str:
    inertia = _table("mca_inertia")
    fig = go.Figure(
        go.Bar(
            x=inertia["axis"],
            y=inertia["eta2_class"],
            marker_color=[TEAL if v >= 0.05 else SLATE for v in inertia["eta2_class"]],
            customdata=inertia["inertia_pct"],
            hovertemplate="axis %{x}<br>η² = %{y:.3f}<br>%{customdata:.2f}% of inertia"
            "<extra></extra>",
        )
    )
    return _div(
        _layout(
            fig,
            xaxis={"title": "MCA axis (largest inertia first)"},
            yaxis={"title": "η² with the label"},
        )
    )


def map_chart() -> str:
    X, y = load_processed(PROCESSED_CSV)
    coords = MCA(n_components=2).fit(X).transform(X)
    fig = go.Figure()
    for label, name in [(0, "edible"), (1, "poisonous")]:
        mask = y.to_numpy() == label
        fig.add_scattergl(
            x=coords[mask, 0].round(4),
            y=coords[mask, 1].round(4),
            mode="markers",
            name=name,
            marker={"color": CLASS_COLORS[name], "size": 4, "opacity": 0.45},
            customdata=X.loc[mask, "odor"],
            hovertemplate="odour: %{customdata}<extra>" + name + "</extra>",
        )
    return _div(_layout(fig, xaxis={"title": "Axis 1"}, yaxis={"title": "Axis 2"}, height=460))


def fold_chart() -> str:
    folds = _table("fold_ordering")
    fig = go.Figure()
    for split, color in [("unshuffled (cv=5)", AMBER), ("shuffled", TEAL)]:
        sub = folds[folds["split"] == split]
        fig.add_box(
            x=sub["model"],
            y=100 * sub["accuracy"],
            name=split,
            marker_color=color,
            boxpoints="all",
            pointpos=0,
            jitter=0.3,
        )
    return _div(_layout(fig, boxmode="group", yaxis={"title": "Accuracy per fold (%)"}))


def reference_table() -> str:
    ref = _table("reference_models")
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(r.model)}</td><td>{html.escape(r.features)}</td>"
        f"<td class='num'>{100 * r.cv_accuracy_mean:.2f} ± {100 * r.cv_accuracy_std:.2f}</td>"
        f"<td class='num'>{r.poisonous_called_edible}</td>"
        f"<td class='num'>{r.edible_called_poisonous}</td>"
        "</tr>"
        for r in ref.itertuples()
    )
    return (
        "<div class='scroll'><table><thead><tr><th>Model</th><th>Features</th>"
        "<th>Accuracy (%)</th><th>Poisonous → edible</th><th>Edible → poisonous</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


CSS = """
:root{--bg:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#0d9488;--card:#f8fafc}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0b1120;--ink:#e2e8f0;
--muted:#94a3b8;--line:#1e293b;--accent:#2dd4bf;--card:#111827}}
:root[data-theme="dark"]{--bg:#0b1120;--ink:#e2e8f0;--muted:#94a3b8;--line:#1e293b;
--accent:#2dd4bf;--card:#111827}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.65 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:48px 16px 80px}
h1{font-size:2rem;line-height:1.2;margin:0 0 8px;letter-spacing:-0.02em}
h2{font-size:1.3rem;margin:48px 0 8px}
.lede{color:var(--muted);font-size:1.1rem;margin:0 0 24px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi b{display:block;font-size:1.6rem;color:var(--accent);font-variant-numeric:tabular-nums}
.kpi span{color:var(--muted);font-size:.9rem}
.chart{border:1px solid var(--line);border-radius:10px;padding:8px;margin:16px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
a{color:var(--accent)}
footer{color:var(--muted);font-size:.9rem;margin-top:56px;border-top:1px solid var(--line);
padding-top:16px}
"""


def build() -> None:
    r = json.loads(RESULTS_JSON.read_text())

    def pct(v: float) -> str:
        return f"{100 * v:.1f}%"

    plotly_js = (
        "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@"
        f"{plotly.offline.get_plotlyjs_version()}/plotly.min.js"
    )
    body = f"""
<h1>What does an MCA keep, and lose, on the UCI Mushroom data?</h1>
<p class="lede">{r["n_specimens"]:,} specimens, {r["n_variables"]} categorical descriptors,
one binary label. The data are almost perfectly separable, so the interesting question is
not <em>can we classify</em> but <em>what a five-axis MCA preserves, and which models can
read it</em>. Every score below is out-of-fold, with the MCA refitted inside each fold.</p>
<div class="kpis">
<div class="kpi"><b>{pct(r["odour_accuracy"])}</b><span>odour alone, with
{r["odour_poisonous_called_edible"]} poisonous called edible</span></div>
<div class="kpi"><b>{pct(r["lda_5_axes"])}</b><span>LDA on 5 MCA axes</span></div>
<div class="kpi"><b>{pct(r["rf_5_axes"])}</b><span>random forest on the same 5 axes</span></div>
<div class="kpi"><b>{r["tree_perfect_leaves"]} leaves</b><span>a depth-{r["tree_perfect_depth"]}
tree on one-hot is perfect</span></div>
</div>

<h2>Accuracy against the number of axes kept</h2>
<p>With five axes, the non-linear readers reach {pct(r["knn_5_axes"])} (k-NN) and
{pct(r["rf_5_axes"])} (random forest). LDA never exceeds {pct(r["lda_max_axes_1_9"])} with up
to nine axes and needs {r["lda_axes_for_99"]} to reach 99%: the information is there, but not
along a straight line.</p>
<div class="chart">{axis_curve_chart(r)}</div>

<h2>Which axes carry the label</h2>
<p>η² is the share of an axis' variance explained by the edible/poisonous label. Axis 1 has
η² = {r["eta2_axis1"]:.2f}; axes 2 to 9 never exceed {r["eta2_axes_2_9_max"]:.2f}, while axis
{r["eta2_best_minor_axis"]}, one of the smallest, reaches {r["eta2_best_minor_value"]:.2f}. Ordering
axes by inertia is not ordering them by usefulness.</p>
<div class="chart">{eta_chart()}</div>
<p>The first five axes hold {r["inertia_axes_1_5_pct"]:.1f}% of the total inertia
({r["benzecri_axes_1_5_pct"]:.1f}% after Benzécri's correction).</p>
<div class="chart">{map_chart()}</div>

<h2>The error that matters</h2>
<p>For a forager, calling a poisonous mushroom edible is the only costly mistake. Odour alone
is {pct(r["odour_accuracy"])} accurate, and all {r["odour_poisonous_called_edible"]} of its
errors are of that kind (odourless poisonous specimens).</p>
{reference_table()}

<h2>Why the earlier version of this project was wrong</h2>
<p>The first version used scikit-learn's default <code>cv=5</code>, which does not shuffle.
The UCI file is sorted in long same-class runs, so each fold saw a different slice of the
catalogue. The spread across folds was read as an unstable model, and the random forest's
train/test gap as overfitting. Re-running that protocol on the current pipeline gives LDA
{pct(r["unshuffled_lda_mean"])} ± {pct(r["unshuffled_lda_std"])} and random forest
{pct(r["unshuffled_rf_mean"])} ± {pct(r["unshuffled_rf_std"])}. With shuffled stratified
folds the LDA spread drops to ±{pct(r["shuffled_lda_std"])} and the random forest scores
{pct(r["shuffled_rf_mean"])}.</p>
<div class="chart">{fold_chart()}</div>

<h2>Limitations</h2>
<p>The records are hypothetical specimens generated from a field guide's descriptions of 23
species, not field observations; random folds therefore test interpolation within known
species, not recognition of a new one. Nothing here is advice on what to eat.</p>

<footer>Code, data provenance and tests: <a href="{REPO}">{REPO.removeprefix("https://")}</a>.
Data: UCI Machine Learning Repository, Mushroom (CC BY 4.0).<br>
Built by <a href="https://github.com/Pchambet">Pierre Chambet</a>, decision science for
operations under uncertainty.</footer>
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mushroom MCA study</title>
<meta name="description" content="What a multiple correspondence analysis keeps and loses on
the UCI Mushroom data, with honest cross-validation.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
rel="stylesheet">
<script src="{plotly_js}"></script>
<style>{CSS}</style></head>
<body><main>{body}</main></body></html>
"""
    SITE_HTML.parent.mkdir(parents=True, exist_ok=True)
    SITE_HTML.write_text(page)
