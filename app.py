"""College project dashboard. Run: streamlit run app.py

Uses results_hybrid_active and the latest detectors.py. No training is performed.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import torch
import detectors as D

BASE = Path(__file__).resolve().parent
ORDER = ["PatchCore", "PaDiM", "ViT-Patch", "AdaptiveHybrid"]
LABELS = {"AdaptiveHybrid": "Our hybrid", "EqualHybrid": "Equal fusion",
          "GuardedHybrid": "Guarded fusion"}


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (BASE / path).resolve()


@st.cache_data(max_entries=24, show_spinner=False)
def cached_csv(path, stamp):
    return pd.read_csv(path)


def read_csv(path):
    path = Path(path)
    return cached_csv(str(path), (path.stat().st_mtime_ns, path.stat().st_size))


def summarize(frame, order):
    columns = {"image_auroc": "Image AUROC (%)", "pixel_auroc": "Pixel AUROC (%)",
               "precision": "Precision (%)", "recall": "Recall (%)", "f1": "F1 (%)",
               "false_alarm_rate": "False alarms (%)",
               "benchmark_median_ms": "Mean run median (ms)",
               "end_to_end_mean_ms": "Response mean (ms)", "memory_mb": "Model/state (MiB)"}
    available = [key for key in columns if key in frame.columns]
    table = frame.groupby("model")[available].mean().reindex(order).dropna(how="all")
    for key in available:
        if key in ("image_auroc", "pixel_auroc", "precision", "recall", "f1", "false_alarm_rate"):
            table[key] *= 100
    table = table.rename(columns=columns)
    table["Runs"] = frame.groupby("model").size().reindex(table.index)
    table.index = [LABELS.get(name, name) for name in table.index]
    table.index.name = "Model"
    return table


def paired_gains(frame):
    keys = ["category", "seed"]
    columns = keys + ["image_auroc", "pixel_auroc", "recall", "false_alarm_rate", "benchmark_median_ms"]
    pc = frame[frame.model == "PatchCore"][columns]
    hybrid = frame[frame.model == "AdaptiveHybrid"][columns]
    paired = hybrid.merge(pc, on=keys, suffixes=("_hybrid", "_pc"), validate="one_to_one")
    gains = paired[keys].copy()
    for key, label in (("image_auroc", "Image AUROC gain (pp)"),
                       ("pixel_auroc", "Pixel AUROC gain (pp)"),
                       ("recall", "Recall change (pp)"),
                       ("false_alarm_rate", "False alarm change (pp)")):
        gains[label] = 100 * (paired[key+"_hybrid"] - paired[key+"_pc"])
    gains["Inference time ratio"] = paired.benchmark_median_ms_hybrid / paired.benchmark_median_ms_pc
    return gains


def grouped_chart(table, ylabel="Percent", percentage=True):
    """Grouped bars keep AUROC/weight series separate instead of stacking them."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(table))
    width = 0.8 / max(len(table.columns), 1)
    for i, column in enumerate(table.columns):
        ax.bar(x - 0.4 + width*(i+0.5), table[column].to_numpy(), width, label=str(column))
    ax.set_xticks(x, [str(value) for value in table.index])
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def auroc_chart(table, zoom=True):
    """Dots show close scores without implying a zero-based bar length."""
    fig, ax = plt.subplots(figsize=(6.4, max(3.8, len(table)*0.8)))
    y = np.arange(len(table))
    spacing = 0.7 / max(len(table.columns), 1)
    colors = {"PatchCore": "#64748b", "Our hybrid": "#2563eb"}
    finite = table.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    lower, upper = 0.0, 100.0
    if zoom and finite.size:
        padding = max(0.15, float(np.ptp(finite))*0.12)
        lower = max(0.0, float(finite.min())-padding)
        upper = min(100.0, float(finite.max())+padding)
    for i, column in enumerate(table.columns):
        positions = y - 0.35 + spacing*(i+0.5)
        values = table[column].to_numpy(dtype=float)
        ax.scatter(values, positions, s=45, label=str(column),
                   color=colors.get(str(column)), zorder=3, clip_on=False)
        for value, position in zip(values, positions):
            if np.isfinite(value):
                near_right = value > upper-(upper-lower)*0.16
                ax.annotate(f"{value:.3f}", (value, position),
                            xytext=(-6 if near_right else 6, 0), textcoords="offset points",
                            ha="right" if near_right else "left", va="center", fontsize=8)
    ax.set_yticks(y, [str(value) for value in table.index])
    ax.invert_yaxis()
    ax.set_xlim(lower, upper)
    ax.set_xlabel("AUROC (%) — zoomed axis" if zoom else "AUROC (%) — full scale")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def gain_chart(gains, column):
    values = gains.groupby("category")[column].mean()
    values.loc["Overall mean"] = gains[column].mean()
    fig, ax = plt.subplots(figsize=(6.4, max(3.8, len(values)*0.55)))
    y = np.arange(len(values))
    ax.barh(y, values.to_numpy(), color=["#16a34a" if v > 0 else "#dc2626" if v < 0 else "#64748b" for v in values])
    ax.axvline(0, color="#334155", linewidth=1)
    extent = max(float(values.abs().max()), 0.01)
    ax.set_xlim(min(0, float(values.min()))-extent*0.28,
                max(0, float(values.max()))+extent*0.28)
    for position, value in zip(y, values):
        ax.annotate(f"{value:+.3f} pp", (value, position),
                    xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center", fontsize=9)
    ax.set_yticks(y, [str(value) for value in values.index])
    ax.invert_yaxis()
    ax.set_xlabel("Hybrid − PatchCore (percentage points)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def artifact_key(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [(str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)]
    for name in D.BASELINES:
        checkpoint = (path.parent / data["checkpoints"][name]).resolve()
        records.append((str(checkpoint), checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size))
    records.append(hashlib.sha256(Path(D.__file__).read_bytes()).hexdigest())
    return tuple(records)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_live(path, key, device):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, expected in data.get("metadata", {}).get("baseline_hashes", {}).items():
        checkpoint = Path(path).parent / data["checkpoints"][name]
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{name} checkpoint changed after this hybrid was fitted.")
    return D.AdaptiveHybrid.load(path, device), threading.RLock()


def make_packet(crop, raw_scores, normalized_maps, combined, image_weights,
                map_weights, normalized_scores, thresholds, hybrid_threshold,
                timings=None, truth=None):
    raw_scores = np.asarray(raw_scores, dtype=float)
    normalized_maps = np.asarray(normalized_maps, dtype=np.float32)
    image_weights, map_weights = np.asarray(image_weights), np.asarray(map_weights)
    contributions = image_weights * np.asarray(normalized_scores)
    image_share = contributions / max(float(contributions.sum()), 1e-12)
    # Explain additive evidence in the highest-scoring 5% of hybrid pixels.
    flat = np.asarray(combined).ravel()
    count = max(1, int(np.ceil(0.05 * len(flat))))
    hotspot = np.argpartition(flat, len(flat)-count)[-count:]
    region_evidence = normalized_maps.reshape(3, -1)[:, hotspot].mean(1)
    heat_contributions = map_weights * region_evidence
    heat_share = heat_contributions / max(float(heat_contributions.sum()), 1e-12)
    maps = {name: normalized_maps[i] for i, name in enumerate(D.BASELINES)}
    maps["AdaptiveHybrid"] = np.asarray(combined, dtype=np.float32)
    scores = {name: float(raw_scores[i]) for i, name in enumerate(D.BASELINES)}
    scores["AdaptiveHybrid"] = float(contributions.sum())
    thresholds = dict(thresholds); thresholds["AdaptiveHybrid"] = float(hybrid_threshold)
    weights = pd.DataFrame({"Model": list(D.BASELINES),
                            "Image weight (%)": 100*image_weights,
                            "Heatmap weight (%)": 100*map_weights,
                            "Weighted image evidence": contributions,
                            "Image contribution (%)": 100*image_share,
                            "Hotspot contribution (%)": 100*heat_share}).set_index("Model")
    return {"crop": crop, "maps": maps, "scores": scores, "thresholds": thresholds,
            "weights": weights, "timings": timings or {}, "truth": truth}


def inspect_live(image, path, device):
    hybrid, lock = load_live(str(path), artifact_key(path), device)
    with lock:
        preprocess_start = time.perf_counter()
        crop = D.crop_view(image)
        x = D.NORMALIZE(crop).unsqueeze(0)
        preprocess_ms = 1000*(time.perf_counter()-preprocess_start)
        for expert in hybrid.experts.values():
            expert.score_tensor(x)  # Warm-up is excluded from displayed timing.
        D.synchronize(hybrid.device)
        start_all = time.perf_counter()
        outputs, timings = [], {}
        for name, expert in hybrid.experts.items():
            D.synchronize(expert.device)
            start = time.perf_counter()
            outputs.append(expert.score_tensor(x))
            D.synchronize(expert.device)
            timings[name] = 1000*(time.perf_counter()-start)
        raw_scores = np.stack([out[0] for out in outputs], axis=1)
        raw_maps = np.stack([out[1] for out in outputs], axis=1)
        result = hybrid.fusion.combine(raw_scores, raw_maps)
        timings["AdaptiveHybrid"] = 1000*(time.perf_counter()-start_all)
        packet = make_packet(crop, raw_scores[0], result["normalized_maps"][0], result["maps"][0],
                             result["image_weights"][0], result["map_weights"][0],
                             result["normalized_scores"][0],
                             {name: expert.threshold for name, expert in hybrid.experts.items()},
                             hybrid.threshold, timings)
        packet["preprocess_ms"] = preprocess_ms
        packet["mode"] = hybrid.fusion.mode
        return packet


def saved_packet(folder, dataset, index, position):
    path = index["paths"][position]
    image_path = dataset / path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}. Check the dataset folder.")
    rows = read_csv(folder / "per_image.csv")
    matching = rows[rows.path == path]
    if len(matching) != 1:
        raise ValueError("Saved image index and per-image results disagree.")
    row = matching.iloc[0]
    fusion, _ = D.FusionMath.load(folder / "hybrid.json")
    raw_maps = []
    for name in D.BASELINES:
        maps = np.load(folder / index["baseline_maps"][name], mmap_mode="r")
        if len(maps) != len(index["paths"]):
            raise ValueError("Saved heatmap count does not match the image index.")
        raw_maps.append(np.array(maps[position], dtype=np.float32))
    maps = np.stack(raw_maps) / fusion.map_scale[:, None, None]
    hybrid_maps = np.load(folder / index["hybrid_maps"]["AdaptiveHybrid"], mmap_mode="r")
    combined = np.array(hybrid_maps[position], dtype=np.float32)
    image_weights = [row[f"{name}_image_weight"] for name in D.BASELINES]
    map_weights = [row[f"{name}_heatmap_weight"] for name in D.BASELINES]
    packet = make_packet(D.crop_view(image_path),
        [row[f"{name}_raw_score"] for name in D.BASELINES], maps, combined,
        image_weights, map_weights,
        [row[f"{name}_normalized_score"] for name in D.BASELINES],
        {name: row[f"{name}_threshold"] for name in D.BASELINES}, row.hybrid_threshold,
        truth=f"{'Defect' if int(row.label) else 'Normal'} · {row.defect_type}")
    if not np.isclose(packet["scores"]["AdaptiveHybrid"], row.hybrid_score, rtol=1e-5):
        raise ValueError("Saved scores and contributions disagree.")
    packet["mode"] = fusion.mode
    return packet


def heatmap_image(amap, maximum, crop=None, opacity=0.5):
    values = np.clip(np.asarray(amap, dtype=float)/maximum, 0, 1)
    rgb = colormaps["inferno"](values)[..., :3] * 255
    if crop is not None:
        rgb = (1-opacity)*np.asarray(crop, dtype=float) + opacity*rgb
    return rgb.clip(0, 255).astype(np.uint8)


def render_packet(packet):
    st.image(packet["crop"], caption="Image region inspected", width=224)
    if packet["truth"]:
        st.caption("Dataset label: " + packet["truth"])
    if packet.get("mode") == "PatchCore_fallback":
        st.info("This saved hybrid uses the PatchCore fallback. Its weights will reflect that.")
    summary = []
    for name in ORDER:
        score, threshold = packet["scores"][name], packet["thresholds"][name]
        row = {"Model": LABELS.get(name, name), "Verdict": "Anomaly" if score > threshold else "Normal",
               "Score": score, "Threshold": threshold,
               "Score / threshold": score/max(abs(threshold), 1e-12)}
        if packet["timings"]:
            row["Measured inference (ms)"] = packet["timings"][name]
        summary.append(row)
    st.dataframe(pd.DataFrame(summary).set_index("Model").round(3), use_container_width=True)
    st.caption("Scores are model-specific evidence, not probabilities. Verdicts use each model's calibrated image threshold.")
    if packet["timings"]:
        st.caption(f"One warmed pass; excludes model loading, image decoding and preprocessing. Shared preprocessing: {packet['preprocess_ms']:.1f} ms. Use the Analytics tab for repeated benchmarks.")
    else:
        st.caption("These are saved experiment predictions. No models were run to display this example.")
    style = st.radio("Heatmap display", ["Overlay", "Heatmap only"], horizontal=True, key="map_style")
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05, key="opacity") if style == "Overlay" else 0.0
    maximum = max(1.0, max(float(amap.max()) for amap in packet["maps"].values()))
    for start in (0, 2):
        columns = st.columns(2)
        for column, name in zip(columns, ORDER[start:start+2]):
            with column:
                st.subheader(LABELS.get(name, name))
                st.image(heatmap_image(packet["maps"][name], maximum,
                                      packet["crop"] if style == "Overlay" else None, opacity), width=280)
    gradient = (colormaps["inferno"](np.linspace(0, 1, 256))[..., :3]*255).astype(np.uint8)
    st.image(np.repeat(gradient[None], 10, axis=0), width=280,
             caption=f"Shared scale: 0 → {maximum:.2f} normal-reference evidence units")
    st.caption("Continuous heatmaps share one color range. Each baseline map is divided by its fitted normal-reference scale. The hybrid combines those maps; no boundaries or binary masks are displayed.")
    st.subheader("How much each model contributes")
    st.dataframe(packet["weights"].round(2), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.caption("Weight assigned to each model")
        grouped_chart(packet["weights"][["Image weight (%)", "Heatmap weight (%)"]])
    with b:
        st.caption("Share of the combined evidence")
        grouped_chart(packet["weights"][["Image contribution (%)", "Hotspot contribution (%)"]])
    st.caption("Weights allocate influence. Image contributions are weight × normalized image score. Hotspot contributions use the highest-scoring 5% of hybrid heatmap pixels. These shares are not probabilities or measured information gain.")
    st.download_button("Download inspection summary", pd.DataFrame(summary).to_csv(index=False),
                       file_name="inspection_summary.csv", mime="text/csv")
    st.download_button("Download model contributions", packet["weights"].to_csv(),
                       file_name="inspection_weights.csv", mime="text/csv")


def analytics(results, category, seed):
    file = results / "metrics.csv"
    if not file.is_file():
        st.info("Choose the completed results_hybrid_active folder to see experiment analytics.")
        return
    frame = read_csv(file)
    required = {"category", "seed", "model", "image_auroc", "pixel_auroc", "benchmark_median_ms", "recall", "false_alarm_rate"}
    if not required.issubset(frame.columns):
        st.error("The metrics file is missing columns from the hybrid experiment.")
        return
    if category != "All categories":
        frame = frame[frame.category == category]
    if seed != "All seeds":
        frame = frame[frame.seed == seed]
    if frame.empty:
        st.info("No completed runs match these filters.")
        return
    controls = st.checkbox("Include equal and guarded fusion controls", value=False)
    order = ORDER + (["EqualHybrid", "GuardedHybrid"] if controls else [])
    selected = frame[frame.model.isin(order)]
    st.subheader("Experiment comparison")
    st.caption("Unweighted means over the selected category/seed runs. All results describe the evaluated dataset; weights were fitted without real test labels.")
    st.dataframe(summarize(selected, order).round(2), use_container_width=True)
    gains = paired_gains(frame)
    if not gains.empty:
        a, b, c = st.columns(3)
        a.metric("Image AUROC gain vs PatchCore", f"{gains['Image AUROC gain (pp)'].mean():+.3f} pp")
        b.metric("Pixel AUROC gain vs PatchCore", f"{gains['Pixel AUROC gain (pp)'].mean():+.3f} pp")
        c.metric("Mean paired inference time ratio", f"{gains['Inference time ratio'].mean():.2f}×")
        st.caption("A time ratio above 1 means the hybrid takes longer. Model/state storage excludes peak inference memory.")
    focus = st.checkbox("Focus charts on PatchCore and our hybrid", value=True)
    zoom = st.checkbox("Zoom AUROC axes to show small differences", value=True)
    chart_frame = selected[selected.model.isin(["PatchCore", "AdaptiveHybrid"])] if focus else selected
    st.caption("Dots are labelled with exact AUROC values. Zoomed axes do not start at zero; use the full-scale option for context. Values and experiment results are unchanged.")
    a, b = st.columns(2)
    with a:
        st.subheader("Image AUROC by category (%)")
        chart = chart_frame.groupby(["category", "model"]).image_auroc.mean().unstack()*100
        auroc_chart(chart.reindex(columns=[name for name in order if name in chart.columns]).rename(columns=LABELS), zoom)
    with b:
        st.subheader("Pixel AUROC by category (%)")
        chart = chart_frame.groupby(["category", "model"]).pixel_auroc.mean().unstack()*100
        auroc_chart(chart.reindex(columns=[name for name in order if name in chart.columns]).rename(columns=LABELS), zoom)
    if not gains.empty:
        st.subheader("Improvement over PatchCore")
        st.caption("Positive values favour the hybrid; negative values favour PatchCore. Differences are percentage points, not relative percentage improvements. These charts show means, not statistical significance.")
        a, b = st.columns(2)
        with a:
            st.write("Image AUROC change")
            gain_chart(gains, "Image AUROC gain (pp)")
        with b:
            st.write("Pixel AUROC change")
            gain_chart(gains, "Pixel AUROC gain (pp)")
        st.subheader("Hybrid gains and tradeoffs by category")
        st.dataframe(gains.groupby("category").mean(numeric_only=True).drop(columns="seed", errors="ignore").round(3), use_container_width=True)
        with st.expander("Individual seed comparisons"):
            st.dataframe(gains.round(3), use_container_width=True)
    file = results / "per_defect.csv"
    if file.is_file():
        defects = read_csv(file)
        if category != "All categories":
            defects = defects[defects.category == category]
        if seed != "All seeds":
            defects = defects[defects.seed == seed]
        st.subheader("Defect-type comparison")
        metric = st.selectbox("Defect metric", ["image_auroc", "pixel_auroc", "recall", "f1"],
                              format_func=lambda value: {"image_auroc":"Image AUROC", "pixel_auroc":"Pixel AUROC", "recall":"Recall", "f1":"F1"}[value])
        table = defects[defects.model.isin(order)].pivot_table(index=["category", "defect_type"], columns="model", values=metric, aggfunc="mean")*100
        if {"PatchCore", "AdaptiveHybrid"}.issubset(table.columns):
            table["Hybrid − PatchCore (pp)"] = table.AdaptiveHybrid - table.PatchCore
        st.dataframe(table.rename(columns=LABELS).round(2), use_container_width=True)
    file = results / "weight_summary.csv"
    if file.is_file():
        weights = read_csv(file)
        if category != "All categories":
            weights = weights[weights.category == category]
        if seed != "All seeds":
            weights = weights[weights.seed == seed]
        st.subheader("Average assigned weights (%)")
        table = weights.pivot_table(index=["category", "model"], columns="head", values="mean_weight", aggfunc="mean")*100
        st.dataframe(table.round(2), use_container_width=True)
    st.download_button("Download filtered metrics", selected.to_csv(index=False),
                       file_name="filtered_metrics.csv", mime="text/csv")


def inspection(results):
    files = sorted(results.glob("*/seed_*/hybrid.json"))
    if not files:
        st.info("No fitted hybrid found. Select the results_hybrid_active folder created by experiments.py.")
        return
    categories = sorted({p.parent.parent.name for p in files})
    a, b, c = st.columns(3)
    category = a.selectbox("Inspection category", categories)
    available = [p for p in files if p.parent.parent.name == category]
    seeds = sorted({int(p.parent.name.removeprefix("seed_")) for p in available})
    seed = b.selectbox("Inspection seed", seeds)
    source = c.selectbox("Image source", ["Upload image", "Saved test image"])
    folder = results / category / f"seed_{seed}"
    identity = (str(results), category, seed, source)
    packet = None
    if source == "Upload image":
        device = st.selectbox("Inference device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
        uploaded = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        st.caption("Choose the matching product category. Inspection covers the resized central crop used during the experiments.")
        if uploaded is not None:
            raw = uploaded.getvalue()
            try:
                identity += (device, hashlib.sha256(raw).hexdigest(), artifact_key(folder / "hybrid.json"))
            except Exception as exc:
                st.error(f"Could not find the fitted hybrid or its baseline checkpoints: {exc}")
                return
            if st.button("Inspect with all four approaches", type="primary"):
                try:
                    with st.spinner("Loading models and inspecting the image…"):
                        image = Image.open(io.BytesIO(raw)).convert("RGB")
                        packet = inspect_live(image, folder / "hybrid.json", device)
                    st.session_state["inspection_result"] = (identity, packet)
                except Exception as exc:
                    st.error(f"Inspection failed: {exc}")
            saved = st.session_state.get("inspection_result")
            if saved and saved[0] == identity:
                packet = saved[1]
    else:
        dataset = local_path(st.text_input("Dataset folder", str(BASE / "datasets" / "MVTecAD")))
        file = folder / "heatmap_index.json"
        if not file.is_file():
            st.info("Saved heatmap index is missing. Use an uploaded image instead.")
            return
        index = json.loads(file.read_text(encoding="utf-8"))
        if not index["paths"]:
            return
        position = st.selectbox("Saved image", range(len(index["paths"])),
                                format_func=lambda i: index["paths"][i])
        try:
            packet = saved_packet(folder, dataset, index, position)
        except Exception as exc:
            st.error(f"Could not display saved example: {exc}")
    if packet is not None:
        render_packet(packet)


def main():
    st.set_page_config(page_title="Adaptive Visual Inspection", layout="wide")
    st.title("Adaptive Visual Inspection")
    st.caption("PatchCore · PaDiM · ViT-Patch · Our adaptive hybrid")
    st.sidebar.header("Experiment results")
    results = local_path(st.sidebar.text_input("Results folder", str(BASE / "results_hybrid_active")))
    categories, seeds = [], []
    if (results / "metrics.csv").is_file():
        metrics = read_csv(results / "metrics.csv")
        if {"category", "seed"}.issubset(metrics.columns):
            categories = sorted(metrics.category.unique())
            seeds = sorted(int(value) for value in metrics.seed.unique())
    category = st.sidebar.selectbox("Analytics category", ["All categories"] + categories)
    seed = st.sidebar.selectbox("Analytics seed", ["All seeds"] + seeds)
    analytics_tab, inspection_tab = st.tabs(["Analytics", "Inspect image / heatmaps"])
    with analytics_tab:
        try:
            analytics(results, category, seed)
        except Exception as exc:
            st.error(f"Could not read experiment analytics: {exc}")
    with inspection_tab:
        inspection(results)
    with st.expander("How the hybrid works"):
        st.write("Each baseline produces an image score and heatmap. Normal-reference scaling makes their evidence comparable. Two mathematical reliability gates, fitted on held-out normal images and synthetic defects, assign separate weights for detection and localization. The active configuration fixes mixing strengths at 0.50 and 1.00; per-model weights still change for each image.")
        st.write("The comparison reports gains as well as losses. An image decision threshold is not a segmentation threshold, and contribution shares are not calibrated confidence probabilities.")


if __name__ == "__main__":
    main()
