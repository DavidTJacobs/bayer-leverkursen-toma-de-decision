"""
Red de pases previa al tiro - Bayer Leverkusen (Bundesliga 2023/2024)

Construye y analiza la red de pases de la misma posesión que conlleva al tiro
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from mplsoccer import Pitch

# =============================================================================
# CONSTANTES
# =============================================================================

POSSESSION_WINDOW_SECONDS = 15  # conservado por compatibilidad, no se usa en get_possession_passes_before_shot
PASS_EVENT_TYPE = "Pass"
PASS_OUTCOME_INCOMPLETE = "Incomplete"
TEAM_NAME = "Bayer Leverkusen"


# =============================================================================
# CONSTRUCCIÓN DE LA RED DE PASES
# =============================================================================



def _resolve_match_id(shot: pd.Series) -> Any:
    """Extrae match_id de una fila de shots_360 (puede tener sufijo _x tras el merge)."""
    for col in ("match_id", "match_id_x", "match_id_y"):
        val = shot.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return val
    return None


def _resolve_match_id_col(df: pd.DataFrame) -> str:
    """Devuelve el nombre de la columna match_id disponible en df_events."""
    for col in ("match_id", "match_id_x"):
        if col in df.columns:
            return col
    raise KeyError("No se encontró columna match_id en df_events")


def get_possession_passes_before_shot(
    shot: pd.Series,
    df_events: pd.DataFrame,
    window_seconds: float = POSSESSION_WINDOW_SECONDS,  # ignorado: se usa posesión completa
) -> pd.DataFrame:
    """
    Extrae los pases completados del Bayer en la misma fase de ataque que el tiro.

    Implementa Ramos et al. (2018): la ventana es la fase de ataque completa,
    definida como el conjunto de eventos desde la recuperación del balón (inicio
    de la possession StatsBomb) hasta el tiro. El campo `possession` de StatsBomb
    es el delimitador natural de cada fase de ataque/defensa.

    No se aplica límite temporal adicional: la posesión ya actúa como frontera
    semántica equivalente a "ball recovery → shot or ball loss".
    """
    shot_match_id = _resolve_match_id(shot)
    match_id_col = _resolve_match_id_col(df_events)
    shot_possession = shot.get("possession")
    shot_event_index = shot.get("event_index")

    if shot_possession is None:
        return pd.DataFrame()

    mask = (
        (df_events[match_id_col] == shot_match_id) &
        (df_events["possession"] == shot_possession) &
        (df_events["type.name"] == PASS_EVENT_TYPE) &
        (df_events["team.name"] == TEAM_NAME) &
        (df_events["event_index"] < shot_event_index)
    )
    candidates = df_events[mask].copy()

    # Excluir pases incompletos
    if "pass.outcome.name" in candidates.columns:
        candidates = candidates[
            candidates["pass.outcome.name"].isna() |
            (candidates["pass.outcome.name"] != PASS_OUTCOME_INCOMPLETE)
        ]

    return candidates


def build_passing_network(passes_df: pd.DataFrame) -> nx.DiGraph:
    """
    Construye una red de pases dirigida y ponderada.

    Nodos: jugadores con posición media de sus pases (red espacialmente embebida).
    Aristas: pases A→B con peso = número de pases entre ese par.
    """
    G = nx.DiGraph()

    if passes_df.empty:
        return G

    for _, row in passes_df.iterrows():
        passer = row.get("player.name")
        recipient = row.get("pass.recipient.name")

        if pd.isna(passer) or pd.isna(recipient):
            continue

        loc = row.get("location")
        x, y = (float(loc[0]), float(loc[1])) if isinstance(loc, list) and len(loc) >= 2 else (60.0, 40.0)
        end_loc = row.get("pass.end_location")
        rx, ry = (
            (float(end_loc[0]), float(end_loc[1]))
            if isinstance(end_loc, list) and len(end_loc) >= 2
            else (x, y)
        )

        if passer not in G.nodes:
            G.add_node(passer, x=x, y=y, n_passes=0)
        n = G.nodes[passer]["n_passes"]
        G.nodes[passer]["x"] = (G.nodes[passer]["x"] * n + x) / (n + 1)
        G.nodes[passer]["y"] = (G.nodes[passer]["y"] * n + y) / (n + 1)
        G.nodes[passer]["n_passes"] = n + 1

        if recipient not in G.nodes:
            G.add_node(recipient, x=rx, y=ry, n_passes=0, n_receives=0)
        r = G.nodes[recipient].get("n_receives", 0)
        G.nodes[recipient]["x"] = (G.nodes[recipient]["x"] * r + rx) / (r + 1)
        G.nodes[recipient]["y"] = (G.nodes[recipient]["y"] * r + ry) / (r + 1)
        G.nodes[recipient]["n_receives"] = r + 1

        if G.has_edge(passer, recipient):
            G[passer][recipient]["weight"] += 1
        else:
            G.add_edge(passer, recipient, weight=1)

    return G


# =============================================================================
# MÉTRICAS DE RED
# =============================================================================

def compute_network_metrics(G: nx.DiGraph) -> dict[str, Any]:
    """
    Métricas de red siguiendo Buldú et al. (2018, 2019) y TEMA_1.

    Macroscala: densidad, clustering, average path length, mayor valor propio.
    Microscala (del tirador): in_degree, betweenness, eigenvector centrality.
    """
    if G.number_of_nodes() == 0:
        return {
            "n_nodes": 0, "n_edges": 0, "total_passes": 0,
            "density": float("nan"), "clustering": float("nan"),
            "avg_path_length": float("nan"), "largest_eigenvalue": float("nan"),
            "in_degree": {}, "out_degree": {}, "betweenness": {}, "eigenvector": {},
        }

    total_passes = sum(d["weight"] for _, _, d in G.edges(data=True))

    # Macroscala
    density = nx.density(G)
    clustering = nx.average_clustering(G.to_undirected(), weight="weight")

    G_dist = G.copy()
    for _, _, d in G_dist.edges(data=True):
        d["distance"] = 1.0 / d["weight"] if d["weight"] > 0 else float("inf")

    try:
        largest_wcc = max(nx.weakly_connected_components(G_dist), key=len)
        avg_path = nx.average_shortest_path_length(
            G_dist.subgraph(largest_wcc), weight="distance"
        )
    except Exception:
        avg_path = float("nan")

    try:
        adj = nx.to_numpy_array(G, weight="weight")
        largest_eigenvalue = float(np.max(np.real(np.linalg.eigvals(adj))))
    except Exception:
        largest_eigenvalue = float("nan")

    # Microscala
    in_degree  = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    try:
        betweenness = nx.betweenness_centrality(G_dist, weight="distance", normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes}

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes}

    return {
        "n_nodes":            G.number_of_nodes(),
        "n_edges":            G.number_of_edges(),
        "total_passes":       total_passes,
        "density":            density,
        "clustering":         clustering,
        "avg_path_length":    avg_path,
        "largest_eigenvalue": largest_eigenvalue,
        "in_degree":          in_degree,
        "out_degree":         out_degree,
        "betweenness":        betweenness,
        "eigenvector":        eigenvector,
    }


def get_shooter_centrality(shooter: str, metrics: dict[str, Any]) -> dict[str, float | None]:
    """Centralidad del tirador dentro de la red de pases previa."""
    def get(d: dict, k: str) -> float | None:
        v = d.get(k)
        return float(v) if v is not None else None

    return {
        "shooter_in_degree":   get(metrics.get("in_degree", {}),   shooter),
        "shooter_out_degree":  get(metrics.get("out_degree", {}),  shooter),
        "shooter_betweenness": get(metrics.get("betweenness", {}), shooter),
        "shooter_eigenvector": get(metrics.get("eigenvector", {}), shooter),
    }


# =============================================================================
# ANÁLISIS DE TODOS LOS TIROS
# =============================================================================

def analyze_passing_networks(
    shots_360: pd.DataFrame,
    df_events: pd.DataFrame,
    window_seconds: float = POSSESSION_WINDOW_SECONDS,
) -> pd.DataFrame:
    """Construye la red de pases previa a cada tiro y devuelve métricas por tiro."""
    records = []

    for _, shot in shots_360.iterrows():
        passes = get_possession_passes_before_shot(shot, df_events, window_seconds)
        G = build_passing_network(passes)
        metrics = compute_network_metrics(G)
        shooter = shot.get("player.name", "")
        centrality = get_shooter_centrality(shooter, metrics)

        record: dict[str, Any] = {
            "shot_id":               shot["id"],
            "n_pass_nodes":          metrics["n_nodes"],
            "n_pass_edges":          metrics["n_edges"],
            "n_passes_prev":         metrics["total_passes"],
            "pass_density":          metrics["density"],
            "pass_clustering":       metrics.get("clustering",         float("nan")),
            "pass_avg_path_length":  metrics.get("avg_path_length",    float("nan")),
            "pass_largest_eigenvalue": metrics.get("largest_eigenvalue", float("nan")),
        }
        record.update(centrality)
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# VISUALIZACIÓN — figura metodológica
# =============================================================================

def plot_passing_network(
    G: nx.DiGraph,
    title: str,
    output_path: Path,
    shot_location: tuple[float, float] | None = None,
    shooter_name: str | None = None,
) -> None:
    """
    Red de pases sobre campo StatsBomb.
    Tamaño de nodo ∝ eigenvector centrality. Grosor de arista ∝ peso.
    Color de arista: azul=adelante, rojo=atrás, gris=lateral.
    Si se proporciona shot_location, marca la posición del tiro y una flecha a portería.
    """
    if G.number_of_nodes() == 0:
        return

    pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white")
    fig, ax = pitch.draw(figsize=(12, 8))  # type: ignore[misc]

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 1.0 for n in G.nodes}

    eig_vals = np.array(list(eig.values()))
    eig_range = eig_vals.max() - eig_vals.min() or 1.0
    node_sizes = {n: 300 + 900 * (eig[n] - eig_vals.min()) / eig_range for n in G.nodes}

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    w_max = max(weights) if weights else 1
    edge_widths = [1 + 7 * (w / w_max) for w in weights]

    pos = {n: (G.nodes[n].get("x", 60.0), G.nodes[n].get("y", 40.0)) for n in G.nodes}

    for (u, v), width in zip(G.edges(), edge_widths):
        dx = pos[v][0] - pos[u][0]
        color = "#2196F3" if dx > 2 else ("#F44336" if dx < -2 else "#9E9E9E")
        ax.annotate(  # type: ignore[misc]
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=width,
                            connectionstyle="arc3,rad=0.1"),
        )

    for n in G.nodes:
        x, y = pos[n]
        ax.scatter(x, y, s=node_sizes[n], color="#FFC107",  # type: ignore[misc]
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(x, y - 3.5, n.split()[-1],  # type: ignore[misc]
                ha="center", va="top", fontsize=7, color="white",
                fontweight="bold", zorder=6)

    # --- Posición del tiro y flecha a portería ---
    if shot_location is not None:
        sx, sy = shot_location
        goal_center = (120.0, 40.0)

        # Flecha del tiro hacia portería
        ax.annotate(  # type: ignore[misc]
            "", xy=goal_center, xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5,
                            linestyle="dashed", connectionstyle="arc3,rad=0.0"),
            zorder=7,
        )

        # Nodo del tirador
        ax.scatter(sx, sy, s=500, color="#E53935", edgecolors="white",  # type: ignore[misc]
                   linewidths=2, zorder=8, marker="*")

        label = shooter_name.split()[-1] if shooter_name else "Tiro"
        ax.text(sx, sy - 4, label,  # type: ignore[misc]
                ha="center", va="top", fontsize=7.5, color="white",
                fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="#E53935", ec="none", alpha=0.7))

    ax.set_title(title, fontsize=10, pad=10)  # type: ignore[misc]
    legend_elements = [
        Line2D([0], [0], color="#2196F3", lw=2, label="Pase adelante"),
        Line2D([0], [0], color="#F44336", lw=2, label="Pase atrás"),
        Line2D([0], [0], color="#9E9E9E", lw=2, label="Pase lateral"),
        Line2D([0], [0], color="white", lw=2, linestyle="dashed", label="Tiro a portería"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7)  # type: ignore[misc]
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example_passing_network(
    shots_360: pd.DataFrame,
    df_events: pd.DataFrame,
    output_dir: Path,
    window_seconds: float = POSSESSION_WINDOW_SECONDS,
) -> None:
    """Guarda la red de pases del tiro con más conexiones previas, con la posición del tiro marcada."""
    best_shot, best_G, best_n = None, nx.DiGraph(), 0

    for _, shot in shots_360.iterrows():
        passes = get_possession_passes_before_shot(shot, df_events, window_seconds)
        G = build_passing_network(passes)
        if G.number_of_edges() > best_n:
            best_n = G.number_of_edges()
            best_shot = shot
            best_G = G

    if best_shot is None or best_G.number_of_nodes() == 0:
        return

    player = best_shot.get("player.name", "")
    minute = int(best_shot.get("minute", 0))
    title = (
        f"Red de pases previa al tiro — {player} (min. {minute})\n"
        f"Fase de ataque completa (Ramos et al., 2018) | "
        f"{best_G.number_of_nodes()} jugadores, {best_G.number_of_edges()} conexiones"
    )

    loc = best_shot.get("location")
    shot_location = (float(loc[0]), float(loc[1])) if isinstance(loc, list) and len(loc) >= 2 else None

    plot_passing_network(
        best_G, title, output_dir / "red_pases_ejemplo.png",
        shot_location=shot_location,
        shooter_name=player,
    )


# =============================================================================
# OUTPUTS PARA LA MEMORIA
# =============================================================================

def save_network_outputs(
    df_network: pd.DataFrame,
    df_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Genera únicamente los outputs relevantes para la memoria:

    1. red_pases_ejemplo.png       — figura metodológica (generada antes)
    2. network_metrics.csv         — métricas de red por tiro (mergeable con decision_results)
    3. network_correlations.csv    — correlaciones red ↔ decisión/xG
    4. network_player_summary.csv  — agregado por jugador para el ranking
    """
    merged = df_results.merge(df_network, on="shot_id", how="left")

    # --- 1. network_metrics.csv ---
    df_network.to_csv(output_dir / "network_metrics.csv", index=False, encoding="utf-8-sig")

    # --- 2. network_correlations.csv ---
    network_cols = [
        "n_passes_prev", "pass_density", "pass_clustering",
        "pass_avg_path_length", "pass_largest_eigenvalue",
        "shooter_in_degree", "shooter_out_degree",
        "shooter_betweenness", "shooter_eigenvector",
    ]
    target_cols = ["good_decision", "decision_value", "shot_xg"]
    corr_df = merged[network_cols + target_cols].copy()
    corr_df["good_decision"] = corr_df["good_decision"].astype(float)
    corr = corr_df.corr(numeric_only=True)[target_cols].loc[network_cols]
    corr.to_csv(output_dir / "network_correlations.csv", encoding="utf-8-sig")

    # --- 3. network_player_summary.csv ---
    player_summary = (
        merged.groupby("player")
        .agg(
            n_shots=("shot_id", "count"),
            good_decision_rate=("good_decision", "mean"),
            avg_passes_prev=("n_passes_prev", "mean"),
            avg_clustering=("pass_clustering", "mean"),
            avg_shooter_betweenness=("shooter_betweenness", "mean"),
            avg_shooter_eigenvector=("shooter_eigenvector", "mean"),
        )
        .reset_index()
        .sort_values("good_decision_rate", ascending=False)
    )
    player_summary.to_csv(output_dir / "network_player_summary.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def run_network_analysis(
    shots_360: pd.DataFrame,
    df_events: pd.DataFrame,
    df_results: pd.DataFrame,
    output_dir: Path,
    window_seconds: float = POSSESSION_WINDOW_SECONDS,
) -> pd.DataFrame:
    """Ejecuta el análisis completo de redes y guarda los outputs para la memoria."""
    print("\nAnalizando redes de pases (Ramos et al. 2018: fase de ataque completa)...")

    plot_example_passing_network(shots_360, df_events, output_dir, window_seconds)

    df_network = analyze_passing_networks(shots_360, df_events, window_seconds)

    n_with_passes = int((df_network["n_passes_prev"] > 0).sum())
    print(f"Tiros analizados: {len(df_network)} | Con pases previos: {n_with_passes}")

    save_network_outputs(df_network, df_results, output_dir)
    print("Outputs de red guardados.")

    return df_network
