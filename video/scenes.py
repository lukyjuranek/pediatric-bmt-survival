"""Eight-scene explainer for the pediatric bone-marrow survival lab.

Rendered with Manim Community. Timings come from video/audio/sceneN.json.
Plot geometry comes from video/chart_data.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Arrow,
    DashedLine,
    DOWN,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    LEFT,
    Line,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    UP,
    VGroup,
    VMobject,
    config,
)

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "chart_data.json").read_text())
S = DATA["summary"]

config.background_color = "#0E1116"
config.media_dir = str(ROOT / "media")

BG = "#0E1116"
ACCENT = "#E3A008"
FG = "#F4F1EA"
MUTED = "#9AA3AD"
DIM = "#343C46"
CARD = "#161B22"
CODE = "Menlo"

CLUSTER_COLORS = ["#F4F1EA", "#E3A008", "#B7B1A6", "#C9842A", "#7E8792"]


def T(text, size=28, color=FG, font=None, line_spacing=0.9):
    kwargs = {"font_size": size, "color": color, "line_spacing": line_spacing}
    if font:
        kwargs["font"] = font
    return Text(text, **kwargs)


def pstr(value: float) -> str:
    if value < 0.05:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return f"{value:.2f}"


def fit(mob, max_width=12.8, max_height=6.05, center_y=-0.32):
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    if mob.height > max_height:
        mob.scale_to_fit_height(max_height)
    mob.move_to([0, center_y, 0])
    return mob


def header(scene_id: str, title: str) -> VGroup:
    index = T(f"{scene_id}  /  8", size=20, color=MUTED, font=CODE)
    name = T(title, size=28, color=FG)
    row = VGroup(index, name).arrange(RIGHT, buff=0.35)
    row.to_edge(UP, buff=0.28).to_edge(LEFT, buff=0.45)
    rule = Line([-6.85, 0, 0], [6.85, 0, 0], color=DIM, stroke_width=1.4)
    rule.next_to(row, DOWN, buff=0.14)
    return VGroup(row, rule)


def card(shape: str, subtitle: str, width=3.35, height=1.5, accent=False) -> VGroup:
    box = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.08,
        stroke_color=ACCENT if accent else DIM,
        stroke_width=1.8 if accent else 1.3,
        fill_color=CARD,
        fill_opacity=1,
    )
    shape_t = T(shape, size=30, color=ACCENT if accent else FG, font=CODE)
    sub_t = T(subtitle, size=16, color=MUTED)
    text = VGroup(shape_t, sub_t).arrange(DOWN, buff=0.08)
    if text.width > width - 0.28:
        text.scale_to_fit_width(width - 0.28)
    text.move_to(box)
    return VGroup(box, text)


def chip(text: str, strike=False, size=15) -> VGroup:
    label = T(text, size=size, color=FG, font=CODE)
    bg = RoundedRectangle(
        width=label.width + 0.24,
        height=max(0.38, label.height + 0.14),
        corner_radius=0.06,
        stroke_color=DIM,
        stroke_width=1.2,
        fill_color=CARD,
        fill_opacity=1,
    )
    label.move_to(bg)
    parts = [bg, label]
    if strike:
        parts.append(
            Line(
                bg.get_corner(UP + LEFT) + np.array([0.08, -0.06, 0]),
                bg.get_corner(DOWN + RIGHT) + np.array([-0.08, 0.06, 0]),
                color=ACCENT,
                stroke_width=1.5,
            )
        )
    return VGroup(*parts)


def code_line(text: str, size=18) -> VGroup:
    label = T(text, size=size, color=FG, font=CODE)
    bg = RoundedRectangle(
        width=label.width + 0.4,
        height=label.height + 0.28,
        corner_radius=0.06,
        stroke_width=0,
        fill_color="#10151C",
        fill_opacity=1,
    )
    label.move_to(bg)
    return VGroup(bg, label)


def mapper(xmin, xmax, ymin, ymax, width, height):
    def convert(x, y):
        px = (x - xmin) / (xmax - xmin) * width - width / 2
        py = (y - ymin) / (ymax - ymin) * height - height / 2
        return np.array([px, py, 0.0])

    return convert


def axes_frame(width, height, xticks, yticks, convert, xlabel, ylabel):
    origin = convert(xticks[0][0], yticks[0][0])
    x_end = convert(xticks[-1][0], yticks[0][0])
    y_end = convert(xticks[0][0], yticks[-1][0])
    # Use the data corners actually requested by the ranges stored on convert
    # via the first and last tick only when ticks span the axis. Callers pass
    # endpoint ticks.
    parts = [
        Line(origin, x_end, color=MUTED, stroke_width=1.6),
        Line(origin, y_end, color=MUTED, stroke_width=1.6),
    ]
    for value, label in xticks:
        point = convert(value, yticks[0][0])
        parts.append(Line(point + DOWN * 0.06, point + UP * 0.06, color=MUTED, stroke_width=1.4))
        if label:
            parts.append(T(label, size=15, color=MUTED).next_to(point, DOWN, buff=0.1))
    for value, label in yticks:
        point = convert(xticks[0][0], value)
        parts.append(Line(point + LEFT * 0.06, point + RIGHT * 0.06, color=MUTED, stroke_width=1.4))
        if label:
            parts.append(T(label, size=15, color=MUTED).next_to(point, LEFT, buff=0.1))
    if xlabel:
        parts.append(T(xlabel, size=16, color=MUTED).next_to(x_end, DOWN, buff=0.42))
    if ylabel:
        parts.append(T(ylabel, size=16, color=MUTED).next_to(y_end, LEFT, buff=0.55))
    return VGroup(*parts)


def step_curve(series, convert, color, width=2.6):
    points = [convert(t, s) for t, s in zip(series["t"], series["s"])]
    cleaned = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - cleaned[-1]) > 1e-4:
            cleaned.append(point)
    vm = VMobject(stroke_color=color, stroke_width=width)
    vm.set_points_as_corners(cleaned)
    return vm


class Narrated(Scene):
    ID = "0"
    TITLE = ""

    def construct(self):
        meta = json.loads((ROOT / "audio" / f"scene{self.ID}.json").read_text())
        self.cue_starts = [float(v) for v in meta["starts"]]
        self.audio_duration = float(meta["duration"])
        self.head = header(self.ID, self.TITLE)
        self.play_scene()
        gap = self.audio_duration - self.time
        if gap > 1 / 60:
            self.wait(gap)
        print(
            f"scene {self.ID}: video {self.time:.2f}s, audio {self.audio_duration:.2f}s",
            flush=True,
        )

    def play_scene(self):
        raise NotImplementedError

    def at(self, index, *anims, run_time=0.6):
        self.wait_until(self.cue_starts[index])
        if anims:
            self.play(*anims, run_time=run_time)

    def wait_until(self, t):
        gap = t - self.time
        if gap > 1 / 50:
            self.wait(gap)


class Scene1(Narrated):
    ID = "1"
    TITLE = "The question"

    def play_scene(self):
        question = T(
            "Which facts, known on the day of transplant,\n"
            "are tied to how long a child survives?",
            size=34,
        )
        sub = T("Lab II   ·   UCI bone-marrow transplant: children", size=20, color=MUTED)
        opening = fit(VGroup(question, sub).arrange(DOWN, buff=0.3), center_y=-0.15)
        self.play(FadeIn(self.head), FadeIn(opening), run_time=0.8)

        cards = VGroup(
            card("Clustering", "PCA, then K-means", width=3.55, height=1.35, accent=True),
            card("Kaplan-Meier", "survival curves", width=3.55, height=1.35),
            card("Cox regression", "hazard ratios", width=3.55, height=1.35),
        ).arrange(RIGHT, buff=0.28)
        pipe = pipeline_row()
        goal = T("Association at baseline. One public cohort.", size=22, color=ACCENT)
        stack = VGroup(cards, pipe, goal).arrange(DOWN, buff=0.38)
        fit(stack, center_y=-0.45)
        self.at(1, FadeOut(opening), FadeIn(cards), run_time=0.7)
        self.at(2, FadeIn(pipe), run_time=0.7)
        self.at(3, FadeIn(goal), run_time=0.5)


def pipeline_row() -> VGroup:
    boxes = [
        card("bone-marrow.arff", "187 x 37", width=3.3, height=1.15),
        card("run_analysis", "pipeline.py", width=3.3, height=1.15, accent=True),
        card("results.json", "figures/  tables/", width=3.3, height=1.15),
    ]
    arrows = []
    row_parts = [boxes[0]]
    for nxt in boxes[1:]:
        arrows.append(Arrow(LEFT, RIGHT, color=ACCENT, stroke_width=2.2, buff=0).scale(0.35))
        row_parts.append(arrows[-1])
        row_parts.append(nxt)
    return VGroup(*row_parts).arrange(RIGHT, buff=0.16)


class Scene2(Narrated):
    ID = "2"
    TITLE = "The cohort"

    def play_scene(self):
        filename = T("bone-marrow.arff", size=24, font=CODE)
        matrix = card("(187, 37)", "validate()", width=3.5, height=1.5, accent=True)
        expected = code_line("EXPECTED_N, P, EVENTS = 187, 37, 85", size=15)
        left = VGroup(filename, matrix, expected).arrange(DOWN, buff=0.2)
        opening = left.copy()
        fit(opening, center_y=-0.15)
        self.play(FadeIn(self.head), FadeIn(opening), run_time=0.7)

        clock = T("survival_time    days from transplant", size=20, font=CODE)
        event = T("survival_status   1 died     0 censored", size=20, font=CODE)
        line = timeline()
        censor = T("Censoring: alive at last contact, then unseen.", size=18, color=MUTED)
        right = VGroup(clock, event, line, censor).arrange(DOWN, buff=0.22)
        split = VGroup(left, right).arrange(RIGHT, buff=0.75)
        fit(split, max_width=12.6)
        self.at(1, FadeOut(opening), FadeIn(split), run_time=0.65)

        counts = VGroup(
            T("85 deaths", size=28, color=ACCENT),
            T("102 censored", size=28),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        counts.next_to(left, DOWN, buff=0.28)
        if counts.get_bottom()[1] < -3.55:
            counts.scale(0.85).next_to(left, DOWN, buff=0.16)
        self.at(2, FadeIn(counts), run_time=0.45)

        plot = overall_km()
        if plot.width > 6.4:
            plot.scale_to_fit_width(6.4)
        plot.move_to(right)
        self.at(3, FadeOut(right), FadeIn(plot), run_time=0.65)


def timeline() -> VGroup:
    base = Line(LEFT * 2.7, RIGHT * 2.7, color=FG, stroke_width=2.2)
    span = 3364 - 6

    def mark(day, label, color=MUTED):
        frac = (day - 6) / span
        point = base.point_from_proportion(frac)
        tick = Line(point + DOWN * 0.1, point + UP * 0.1, color=color, stroke_width=2)
        text = T(label, size=15, color=color).next_to(tick, DOWN, buff=0.12)
        return VGroup(tick, text)

    return VGroup(
        base,
        mark(6, "day 6"),
        mark(1522, "follow-up 1522", ACCENT),
        mark(3364, "day 3364"),
    )


def overall_km() -> VGroup:
    convert = mapper(0, 3364, 0, 1.05, 5.4, 3.3)
    frame = axes_frame(
        5.4,
        3.3,
        [(0, "0"), (1522, "1522"), (3364, "3364")],
        [(0, "0"), (0.5, "0.5"), (1, "1")],
        convert,
        "days",
        "S(t)",
    )
    curve = step_curve(DATA["km_all"], convert, ACCENT, width=2.8)
    half = DashedLine(convert(0, 0.5), convert(3364, 0.5), color=DIM, stroke_width=1.3)
    note = T("median survival not reached", size=18, color=ACCENT)
    note.next_to(frame, UP, buff=0.15)
    follow = Dot(convert(1522, 0.5), radius=0.05, color=ACCENT)
    return VGroup(frame, half, curve, follow, note)


class Scene3(Narrated):
    ID = "3"
    TITLE = "What is allowed in the model"

    def play_scene(self):
        banned_names = [
            "Relapse",
            "IIIV",
            "aGvHDIIIIV",
            "extcGvHD",
            "ANCrecovery",
            "PLTrecovery",
            "time_to_aGvHD_III_IV",
        ]
        kept_names = [
            "Recipientage",
            "Donorage",
            "CD34kgx10d6",
            "Riskgroup",
            "Disease",
            "HLAmatch",
            "Stemcellsource",
        ]
        banned = VGroup(*[chip(name, strike=True, size=15) for name in banned_names]).arrange(
            DOWN, buff=0.1, aligned_edge=LEFT
        )
        kept = VGroup(*[chip(name, size=15) for name in kept_names]).arrange(
            DOWN, buff=0.1, aligned_edge=LEFT
        )
        left_title = T("POST_TRANSPLANT", size=20, color=ACCENT, font=CODE)
        right_title = T("baseline, kept", size=20, color=FG, font=CODE)
        left = VGroup(left_title, banned).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        right = VGroup(right_title, kept).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        columns = VGroup(left, right).arrange(RIGHT, buff=1.1, aligned_edge=UP)
        fit(columns, max_height=5.5, center_y=0.15)
        self.play(FadeIn(self.head), FadeIn(columns), run_time=0.7)

        sentinel = VGroup(
            T("SENTINEL = 1_000_000", size=20, color=ACCENT, font=CODE),
            T("ANCrecovery  5     PLTrecovery  17     time_to_aGvHD_III_IV  145", size=16, color=FG, font=CODE),
        ).arrange(DOWN, buff=0.12)
        sentinel.next_to(columns, DOWN, buff=0.28)
        self.at(1, FadeIn(sentinel), run_time=0.5)

        cleaning = VGroup(
            code_line('load_arff()   latin-1,  "?" -> NA', size=16),
            code_line("clean()  adds log_CD34, log_CD3", size=16),
            T("(187, 37)   ->   (187, 39)", size=22, color=ACCENT, font=CODE),
        ).arrange(DOWN, buff=0.12)
        cleaning.move_to(sentinel)
        self.at(2, FadeOut(sentinel), FadeIn(cleaning), run_time=0.55)

        missing = T(
            f"{S['n_missing_rows']} rows with any missing field\n"
            f"{S['n_baseline_missing']} rows missing a baseline field",
            size=22,
        )
        missing.move_to(cleaning)
        self.at(3, FadeOut(cleaning), FadeIn(missing), run_time=0.5)


class Scene4(Narrated):
    ID = "4"
    TITLE = "Two matrices"

    def play_scene(self):
        df = card("(187, 39)", "clean(df)", width=3.2, height=1.35)
        c13 = card("(187, 13)", "13 baseline inputs", width=3.5, height=1.35)
        c17 = card("(187, 17)", "one-hot Disease + scale", width=3.5, height=1.35, accent=True)
        cox = card("(187, 8)", "build_cox_frame", width=3.5, height=1.35, accent=True)
        names = [
            "Recipientage_per5",
            "Donorage_per10",
            "log_CD34",
            "Diseasegroup",
            "Riskgroup",
            "Txpostrelapse",
            "HLAmismatch",
            "Stemcellsource",
        ]
        chips = VGroup(*[chip(name, size=13) for name in names]).arrange_in_grid(
            rows=2, cols=4, buff=(0.12, 0.1)
        )
        vif = T(f"max VIF  {S['vif_max']:.2f}     85 events / 8 covariates", size=22, color=ACCENT)
        rule = T("Both matrices drop POST_TRANSPLANT and the outcomes.", size=18, color=MUTED)

        df.move_to(UP * 1.85)
        c13.move_to(LEFT * 3.35 + DOWN * 0.15)
        c17.move_to(c13)
        cox.move_to(RIGHT * 3.35 + UP * 0.55)
        chips.next_to(cox, DOWN, buff=0.22)
        left_arrow = Arrow(df.get_bottom(), c13.get_top(), buff=0.12, color=ACCENT, stroke_width=2)
        right_arrow = Arrow(df.get_bottom(), cox.get_top(), buff=0.12, color=ACCENT, stroke_width=2)
        vif.next_to(VGroup(c13, chips), DOWN, buff=0.3)
        rule.next_to(vif, DOWN, buff=0.16)

        layout = VGroup(df, left_arrow, right_arrow, c13, c17, cox, chips, vif, rule)
        fit(layout, max_height=5.9)
        for part in list(layout):
            layout.remove(part)
        self.play(FadeIn(self.head), FadeIn(df), FadeIn(rule), run_time=0.7)
        self.at(1, FadeIn(left_arrow), FadeIn(c13), run_time=0.55)
        self.wait_until(self.cue_starts[1] + 0.62 * (self.cue_starts[2] - self.cue_starts[1]))
        self.play(FadeOut(c13), FadeIn(c17), run_time=0.45)
        self.at(2, FadeIn(right_arrow), FadeIn(cox), FadeIn(chips), run_time=0.6)
        self.at(3, FadeIn(vif), run_time=0.4)


class Scene5(Narrated):
    ID = "5"
    TITLE = "PCA, then K-means"

    def play_scene(self):
        scatter = pca_panel()
        opening = fit(scatter, max_width=8.5, max_height=5.4)
        self.play(FadeIn(self.head), FadeIn(opening), run_time=0.7)

        bars = silhouette_panel()
        split = VGroup(scatter.copy(), bars).arrange(RIGHT, buff=0.45)
        fit(split, max_height=5.5)
        code = code_line('PCA(n_components=0.85)   KMeans(n_init=50, random_state=42)', size=15)
        code.next_to(split, DOWN, buff=0.18)
        if code.get_bottom()[1] < -3.7:
            code.scale(0.9).next_to(split, DOWN, buff=0.1)
        self.at(1, FadeOut(opening), FadeIn(split), FadeIn(code), run_time=0.7)

        note = T("K = 6, 7, 8 have min size 9, so they are ineligible. K = 5 wins.", size=20, color=ACCENT)
        note.next_to(code, DOWN, buff=0.12)
        self.at(2, FadeIn(note), run_time=0.45)

        cards = cluster_cards()
        fit(cards, max_height=4.6, center_y=0.05)
        self.at(3, FadeOut(split), FadeOut(code), FadeOut(note), FadeIn(cards), run_time=0.7)

        db = T(
            f"DBSCAN  eps {S['dbscan_eps']:.2f}   min_samples {S['dbscan_min_samples']}"
            f"   clusters {S['dbscan_clusters']}   noise {S['dbscan_noise']}   ARI {S['dbscan_ari']:.2f}",
            size=18,
            color=MUTED,
            font=CODE,
        )
        db.next_to(cards, DOWN, buff=0.28)
        self.at(4, FadeIn(db), run_time=0.45)


def pca_panel() -> VGroup:
    xs = np.array(DATA["pca"]["x"], dtype=float)
    ys = np.array(DATA["pca"]["y"], dtype=float)
    labels = DATA["pca"]["label"]
    pad_x = 0.08 * (xs.max() - xs.min())
    pad_y = 0.08 * (ys.max() - ys.min())
    convert = mapper(xs.min() - pad_x, xs.max() + pad_x, ys.min() - pad_y, ys.max() + pad_y, 5.6, 4.4)
    v1, v2 = S["pca_2d"]
    frame = axes_frame(
        5.6,
        4.4,
        [(xs.min(), f"PC1  {100 * v1:.0f}%"), (xs.max(), "")],
        [(ys.min(), ""), (ys.max(), f"PC2  {100 * v2:.0f}%")],
        convert,
        "",
        "",
    )
    dots = VGroup(
        *[
            Dot(convert(x, y), radius=0.028, color=CLUSTER_COLORS[int(lab)])
            for x, y, lab in zip(xs, ys, labels)
        ]
    )
    title = T(
        f"{S['pca_components']} components, {100 * S['pca_variance']:.0f}% variance",
        size=18,
        color=ACCENT,
    )
    title.next_to(frame, UP, buff=0.12)
    return VGroup(frame, dots, title)


def silhouette_panel() -> VGroup:
    rows = DATA["k_table"]
    max_s = 0.26
    bars = VGroup()
    height = 3.5
    for row in rows:
        chosen = row["k"] == S["k"]
        color = ACCENT if chosen else (MUTED if row["eligible"] else DIM)
        bar = RoundedRectangle(
            width=0.38,
            height=max(0.08, row["silhouette"] / max_s * height),
            corner_radius=0.04,
            stroke_width=0,
            fill_color=color,
            fill_opacity=1,
        )
        label = T(str(row["k"]), size=16, color=FG if chosen else MUTED, font=CODE)
        label.next_to(bar, DOWN, buff=0.08)
        bars.add(VGroup(bar, label))
    bars.arrange(RIGHT, buff=0.16, aligned_edge=DOWN)
    title = T(f"silhouette, chosen K = {S['k']}", size=18, color=FG)
    rule = T("keep min size >= 15", size=16, color=MUTED, font=CODE)
    title.next_to(bars, UP, buff=0.2)
    rule.next_to(title, UP, buff=0.08)
    return VGroup(rule, title, bars)


def cluster_cards() -> VGroup:
    cards = VGroup()
    for row in DATA["clusters"]:
        color = CLUSTER_COLORS[row["id"]]
        name = T(wrap_name(row["name"]), size=16, color=FG, line_spacing=0.85)
        ident = T(f"C{row['id']}", size=22, color=color, font=CODE)
        meta = T(f"n={row['n']}   events={row['events']}", size=14, color=MUTED, font=CODE)
        median = T(
            "median not reached" if row["median"] is None else f"median {row['median']:.0f} d",
            size=14,
            color=ACCENT if row["median"] else MUTED,
        )
        body = VGroup(ident, name, meta, median).arrange(DOWN, buff=0.08)
        box = RoundedRectangle(
            width=2.45,
            height=2.55,
            corner_radius=0.08,
            stroke_color=color,
            stroke_width=1.6,
            fill_color=CARD,
            fill_opacity=1,
        )
        if body.width > 2.2:
            body.scale_to_fit_width(2.2)
        body.move_to(box)
        cards.add(VGroup(box, body))
    return cards.arrange(RIGHT, buff=0.14)


def wrap_name(name: str) -> str:
    wrapped = {
        "ALL, low-risk": "ALL,\nlow-risk",
        "chronic, poorer HLA match": "chronic,\npoorer HLA match",
        "mostly marrow, lower CD34": "mostly marrow,\nlower CD34",
        "AML, often post-relapse": "AML, often\npost-relapse",
        "nonmalignant, younger recipients": "nonmalignant,\nyounger recipients",
    }
    return wrapped.get(name, name)


class Scene6(Narrated):
    ID = "6"
    TITLE = "Kaplan-Meier"

    def play_scene(self):
        plot, curves, cox_curves = km_plot()
        definition = T(
            "Each patient counts until death or the day follow-up stops.",
            size=20,
            color=MUTED,
        )
        block = VGroup(definition, plot).arrange(DOWN, buff=0.2)
        fit(block, max_height=5.7)
        chart = plot[0]
        for curve in (*curves, *cox_curves):
            chart.remove(curve)
        self.play(FadeIn(self.head), FadeIn(definition), FadeIn(plot), run_time=0.7)

        stat = T(
            f"log-rank  {S['logrank_stat']:.2f}     p = {pstr(S['logrank_p'])}",
            size=22,
            color=ACCENT,
            font=CODE,
        )
        stat.next_to(block, DOWN, buff=0.16)
        self.at(1, *[FadeIn(curve) for curve in curves], FadeIn(stat), run_time=0.8)

        callout = T(
            "C2 median 421 d     C0, C1, C4 median not reached",
            size=18,
            color=FG,
        )
        callout.next_to(stat, DOWN, buff=0.1)
        self.at(2, FadeIn(callout), run_time=0.4)
        lrt = T(
            f"model B likelihood ratio  {S['lrt_stat']:.2f} on {S['lrt_df']} df     p = {pstr(S['lrt_p'])}",
            size=18,
            color=MUTED,
            font=CODE,
        )
        lrt.move_to(callout)
        self.at(
            3,
            *[FadeIn(curve) for curve in cox_curves],
            FadeOut(callout),
            FadeIn(lrt),
            run_time=0.7,
        )


def km_plot():
    convert = mapper(0, 3400, 0, 1.05, 8.4, 4.15)
    frame = axes_frame(
        8.4,
        4.15,
        [(0, "0"), (1000, "1000"), (2000, "2000"), (3000, "3000")],
        [(0, "0"), (0.5, "0.5"), (1, "1")],
        convert,
        "days",
        "S(t)",
    )
    curves = []
    dashed = []
    legend_rows = []
    for row in DATA["clusters"]:
        cid = str(row["id"])
        color = CLUSTER_COLORS[row["id"]]
        curves.append(step_curve(DATA["km"][cid], convert, color, width=2.5))
        smooth = step_curve(DATA["cox_curves"][cid], convert, color, width=1.8)
        dashed.append(DashedVMobject(smooth, num_dashes=28, dashed_ratio=0.55))
        swatch = Line(LEFT * 0.22, RIGHT * 0.22, color=color, stroke_width=3)
        legend_rows.append(
            VGroup(swatch, T(f"C{row['id']}  {row['name']}", size=14, color=FG)).arrange(RIGHT, buff=0.12)
        )
    legend_rows.append(
        VGroup(
            DashedLine(LEFT * 0.22, RIGHT * 0.22, color=MUTED, stroke_width=2),
            T("Cox mean", size=14, color=MUTED),
        ).arrange(RIGHT, buff=0.12)
    )
    legend = VGroup(*legend_rows).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
    chart = VGroup(frame, *curves, *dashed)
    group = VGroup(chart, legend).arrange(RIGHT, buff=0.35)
    return group, curves, dashed


class Scene7(Narrated):
    ID = "7"
    TITLE = "The Cox model"

    def play_scene(self):
        formula = T("h(t | x)  =  h0(t)  *  exp(beta * x)", size=32, font=CODE)
        gloss = T("baseline hazard, multiplied by the covariates", size=20, color=MUTED)
        ratio = T("hazard ratio  =  exp(beta)", size=26, font=CODE, color=ACCENT)
        sides = VGroup(
            T("HR  >  1    higher risk", size=22),
            T("HR  <  1    lower risk", size=22),
        ).arrange(RIGHT, buff=0.7)
        intro = fit(VGroup(formula, gloss, ratio, sides).arrange(DOWN, buff=0.28))
        self.play(FadeIn(self.head), FadeIn(intro), run_time=0.8)
        self.at(1, run_time=0.1)

        small = formula.copy().scale(0.62)
        forest = forest_plot()
        small.next_to(forest, UP, buff=0.22)
        board = fit(VGroup(small, forest), max_height=5.5, center_y=-0.05)
        self.at(2, FadeOut(intro), FadeIn(board), run_time=0.7)

        note = T(
            f"Stem-cell source HR {S['stem_hr']:.2f}, p = {pstr(S['stem_p'])}."
            f"   Reported fit stratifies on HLAmismatch.",
            size=18,
            color=MUTED,
        )
        note.next_to(board, DOWN, buff=0.16)
        self.at(3, FadeIn(note), run_time=0.45)

        metrics = T(
            f"train C-index {S['cindex_train']:.2f}    "
            f"5-fold CV {S['cindex_cv_mean']:.2f} ± {S['cindex_cv_std']:.2f}    "
            f"model C penalizer 0.1",
            size=20,
            color=ACCENT,
            font=CODE,
        )
        metrics.next_to(note, DOWN, buff=0.12)
        self.at(4, FadeIn(metrics), run_time=0.45)


def forest_plot() -> VGroup:
    rows = list(reversed(DATA["cox"]))
    lo, hi = np.log10(0.3), np.log10(2.8)
    width = 5.6

    def xpos(hr):
        clipped = min(max(hr, 0.32), 2.7)
        return (np.log10(clipped) - lo) / (hi - lo) * width

    labels = []
    for row in rows:
        color = ACCENT if row["name"] == "Recipientage_per5" else FG
        labels.append(T(row["label"], size=18, color=color))
    label_col = VGroup(*labels).arrange(DOWN, buff=0.22, aligned_edge=RIGHT)
    axis_y = label_col.get_bottom()[1] - 0.45
    origin_x = label_col.get_right()[0] + 0.45
    parts = [label_col]
    baseline = Line(
        [origin_x, axis_y, 0],
        [origin_x + width, axis_y, 0],
        color=DIM,
        stroke_width=1.4,
    )
    parts.append(baseline)
    one = origin_x + xpos(1.0)
    parts.append(Line([one, label_col.get_top()[1] + 0.15, 0], [one, axis_y, 0], color=ACCENT, stroke_width=1.3))
    for value in (0.5, 1, 2):
        x = origin_x + xpos(value)
        parts.append(T(f"{value:g}", size=15, color=MUTED, font=CODE).move_to([x, axis_y - 0.28, 0]))
    for label, row in zip(labels, rows):
        y = label.get_y()
        color = ACCENT if row["name"] == "Recipientage_per5" else MUTED
        x0 = origin_x + xpos(row["lo"])
        x1 = origin_x + xpos(row["hi"])
        parts.append(Line([x0, y, 0], [x1, y, 0], color=color, stroke_width=2))
        parts.append(Line([x0, y - 0.06, 0], [x0, y + 0.06, 0], color=color, stroke_width=1.6))
        parts.append(Line([x1, y - 0.06, 0], [x1, y + 0.06, 0], color=color, stroke_width=1.6))
        parts.append(Dot([origin_x + xpos(row["hr"]), y, 0], radius=0.055, color=color))
        parts.append(
            T(f"{row['hr']:.2f}", size=15, color=color, font=CODE).move_to([origin_x + width + 0.4, y, 0])
        )
    parts.append(T("hazard ratio", size=15, color=MUTED).next_to(baseline, DOWN, buff=0.42))
    return VGroup(*parts)


class Scene8(Narrated):
    ID = "8"
    TITLE = "What to take away"

    def play_scene(self):
        items = [
            takeaway(
                "1",
                f"Recipient age survives the multivariable model.\n"
                f"HR {S['age_hr']:.2f} per 5 years, p = {pstr(S['age_p'])}.",
            ),
            takeaway(
                "2",
                f"Five clusters describe the cohort.\n"
                f"They do not separate survival. Log-rank p = {pstr(S['logrank_p'])}.",
            ),
            takeaway(
                "3",
                f"187 patients, 85 deaths.\n"
                f"Cross-validated C-index {S['cindex_cv_mean']:.2f}.",
            ),
        ]
        stack = VGroup(*items).arrange(DOWN, buff=0.22)
        limits = T(
            f"Clusters fit on all rows.   DBSCAN ARI = {S['dbscan_ari']:.2f}.\n"
            f"Baseline-missing log-rank p = {pstr(S['baseline_missing_p'])}.\n"
            "Predictors frozen on the day of transplant.",
            size=20,
            color=MUTED,
        )
        block = VGroup(stack, limits).arrange(DOWN, buff=0.35)
        fit(block, max_height=5.6)
        for part in (items[0], items[1], items[2], limits):
            block.remove(part) if part in block.submobjects else stack.remove(part)
        self.play(FadeIn(self.head), run_time=0.35)
        self.at(0, FadeIn(items[0]), run_time=0.5)
        self.at(1, FadeIn(items[1]), run_time=0.5)
        self.at(2, FadeIn(items[2]), run_time=0.5)
        self.at(3, FadeIn(limits), run_time=0.5)


def takeaway(number: str, text: str) -> VGroup:
    badge = T(number, size=26, color=ACCENT, font=CODE)
    body = T(text, size=24, line_spacing=0.9)
    row = VGroup(badge, body).arrange(RIGHT, buff=0.3, aligned_edge=UP)
    box = RoundedRectangle(
        width=max(8.5, row.width + 0.6),
        height=row.height + 0.36,
        corner_radius=0.08,
        stroke_color=DIM,
        stroke_width=1.2,
        fill_color=CARD,
        fill_opacity=1,
    )
    row.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.28)
    return VGroup(box, row)
