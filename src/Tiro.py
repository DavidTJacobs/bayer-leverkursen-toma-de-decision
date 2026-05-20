"""
Análisis de toma de decisión en el tiro - Bayer Leverkusen (Bundesliga 2023/2024)
Fuente de datos: StatsBomb Open Data (eventos + 360)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from mplsoccer import Pitch

from passing_network import run_network_analysis


# =============================================================================
# RUTAS
# =============================================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EVENTS_DIR = DATA_DIR / "events"
THREE_SIXTY_DIR = DATA_DIR / "three-sixty"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTES
# =============================================================================

# Portería rival en coordenadas StatsBomb
LEFT_POST = (120, 36)
RIGHT_POST = (120, 44)
GOAL_CENTER = (120, 40)

# Parámetros de la métrica de probabilidad de marcar
LAMBDA_ = 0.08   # Penalización por distancia al gol
ALPHA = 0.6      # Balance entre penalización por defensores y cercanía al más próximo

# Parámetros del modelo de intercepción de pase
SIGMA_BASE = 0.45   # Incertidumbre temporal proporcional a la distancia del pase
N_STEPS = 25        # Discretización de la trayectoria del pase

# Umbrales de clasificación del pase
MAX_INTERCEPTION_PROB = 0.5
MAX_PASS_RISK = 0.0

# Ventanas temporales y estados del marcador para el heatmap
TIME_WINDOW_ORDER = ["0-23", "23-45+", "45-68", "68-90+"]
SCORE_STATE_ORDER = ["-2+", "-1", "0", "+1", "+2+"]


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def load_json(path: Path) -> list | dict:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def load_matches(data_dir: Path) -> pd.DataFrame:
    """Carga y normaliza el archivo de partidos del Bayer Leverkusen."""
    matches = pd.DataFrame(load_json(data_dir / "bayer_leverkusen_matches.json"))
    matches["match_id"] = matches["match_id"].astype(int)
    matches["home_team"] = matches["home_team"].apply(lambda t: t["home_team_name"])
    matches["away_team"] = matches["away_team"].apply(lambda t: t["away_team_name"])
    return matches


def get_valid_match_ids(matches: pd.DataFrame) -> list[int]:
    """Devuelve los match_id que tienen archivos de eventos y 360 disponibles."""
    return [
        mid for mid in matches["match_id"]
        if (EVENTS_DIR / f"{mid}.json").exists()
        and (THREE_SIXTY_DIR / f"{mid}.json").exists()
    ]


def load_all_events_and_360(match_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y concatena eventos y datos 360 de todos los partidos indicados."""
    all_events, all_360 = [], []

    for match_id in match_ids:
        df_events = pd.json_normalize(load_json(EVENTS_DIR / f"{match_id}.json"))
        df_360 = pd.json_normalize(load_json(THREE_SIXTY_DIR / f"{match_id}.json"))

        df_events["match_id"] = match_id
        df_events["event_index"] = range(len(df_events))
        df_360["match_id"] = match_id

        all_events.append(df_events)
        all_360.append(df_360)

    return pd.concat(all_events, ignore_index=True), pd.concat(all_360, ignore_index=True)


# =============================================================================
# MARCADOR EN VIVO
# =============================================================================

def build_goal_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Extrae todos los eventos de gol (tiro y en propia puerta) del DataFrame de eventos."""
    shot_goals = events_df[
        (events_df["type.name"] == "Shot") &
        (events_df["shot.outcome.name"] == "Goal")
    ][["match_id", "event_index", "team.name", "minute", "second"]].copy()

    own_goals = events_df[
        events_df["type.name"] == "Own Goal For"
    ][["match_id", "event_index", "team.name", "minute", "second"]].copy()

    return pd.concat([shot_goals, own_goals], ignore_index=True)


def score_difference_before_event(
    match_id: int,
    event_index: int,
    goal_events: pd.DataFrame,
    team_name: str = "Bayer Leverkusen",
) -> int:
    """Diferencia de goles (equipo - rival) antes del evento indicado."""
    previous = goal_events[
        (goal_events["match_id"] == match_id) &
        (goal_events["event_index"] < event_index)
    ]
    goals_for = (previous["team.name"] == team_name).sum()
    goals_against = (previous["team.name"] != team_name).sum()
    return int(goals_for - goals_against)


# =============================================================================
# CLASIFICADORES DE CONTEXTO
# =============================================================================

def time_window(period: int, minute: int) -> str:
    """Clasifica el minuto del partido en una ventana temporal de 4 tramos."""
    if period == 1:
        return "0-23" if minute < 23 else "23-45+"
    return "45-68" if minute < 68 else "68-90+"


def score_state(score_diff: int) -> str:
    """Convierte la diferencia de goles en una categoría de marcador agrupada."""
    if score_diff <= -2:
        return "-2+"
    if score_diff >= 2:
        return "+2+"
    if score_diff > 0:
        return f"+{score_diff}"
    return str(score_diff)


# =============================================================================
# GEOMETRÍA Y MÉTRICAS DE TIRO
# =============================================================================

def distance(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def as_point(location) -> tuple[float, float]:
    return float(location[0]), float(location[1])


def shot_angle(
    player_pos: tuple,
    left_post: tuple = LEFT_POST,
    right_post: tuple = RIGHT_POST,
) -> float:
    """Ángulo subtendido entre el tirador y los dos postes (grados)."""
    a = distance(player_pos, left_post)
    b = distance(player_pos, right_post)
    c = distance(left_post, right_post)

    if a == 0 or b == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, (a**2 + b**2 - c**2) / (2 * a * b)))
    return math.degrees(math.acos(cos_angle))


def triangle_area(a, b, c) -> float:
    a, b, c = as_point(a), as_point(b), as_point(c)
    return abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0


def point_in_triangle(point, a, b, c, tol: float = 1e-6) -> bool:
    """Comprueba si un punto está dentro del triángulo (a, b, c) usando coordenadas baricéntricas."""
    point, a, b, c = as_point(point), as_point(a), as_point(b), as_point(c)
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < tol:
        return False
    w1 = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / denom
    w2 = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / denom
    w3 = 1 - w1 - w2
    return w1 >= -tol and w2 >= -tol and w3 >= -tol


def defenders_in_triangle(
    player_pos: tuple,
    opponents_df: pd.DataFrame,
) -> pd.DataFrame:
    """Filtra los defensores que están dentro del triángulo de tiro."""
    if opponents_df.empty:
        return opponents_df.copy()
    mask = opponents_df["location"].apply(
        lambda loc: point_in_triangle(loc, player_pos, LEFT_POST, RIGHT_POST)
    )
    return opponents_df[mask].copy()


def nearest_defender_distance(player_pos: tuple, opponents_df: pd.DataFrame) -> float:
    """Distancia al defensor más cercano dentro del triángulo de tiro."""
    in_triangle = defenders_in_triangle(player_pos, opponents_df)
    if in_triangle.empty:
        return float("inf")
    return min(distance(as_point(player_pos), as_point(loc)) for loc in in_triangle["location"])


def distance_to_goal(player_pos: tuple, goal_center: tuple = GOAL_CENTER) -> float:
    return distance(as_point(player_pos), as_point(goal_center))


def scoring_probability(
    player_pos: tuple,
    opponents_df: pd.DataFrame,
    lambda_: float = LAMBDA_,
    alpha: float = ALPHA,
) -> dict:
    """
    Métrica de probabilidad de marcar.

    p = exp( -λ·d₀  -  α·n_def  -  (1-α)/d_m )

    - d₀: distancia al centro de portería
    - n_def: defensores dentro del triángulo de tiro
    - d_m: distancia al defensor más cercano del triángulo
    """
    d_0 = distance_to_goal(player_pos)
    n_def = len(defenders_in_triangle(player_pos, opponents_df))
    d_m = nearest_defender_distance(player_pos, opponents_df)

    if d_m == 0:
        d_m = 1e-6

    p_score = math.exp(
        (-lambda_ * d_0) -
        (alpha * n_def) -
        ((1 - alpha) / d_m)
    )

    return {"d_0": d_0, "n_def": n_def, "d_m": d_m, "p_score": p_score}


# =============================================================================
# MODELO DE INTERCEPCIÓN DE PASE
# =============================================================================

def interpolate(start: tuple, end: tuple, t: float) -> tuple:
    return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)


def time_uncertainty(pass_distance: float) -> float:
    """Incertidumbre temporal modelada como función lineal de la distancia del pase."""
    return SIGMA_BASE * pass_distance


def interception_logistic_probability(
    player_dist: float,
    ball_dist: float,
    pass_distance: float,
) -> float:
    """
    Probabilidad logística de que un jugador intercepte el balón en un punto dado.

    Compara la distancia del jugador al punto con la distancia recorrida por el balón,
    usando una distribución logística con escala proporcional a la incertidumbre temporal.
    """
    sigma_d = time_uncertainty(pass_distance)
    scale = (math.sqrt(3) * sigma_d) / math.pi
    exponent = max(-50.0, min(50.0, (player_dist - ball_dist) / scale))
    return 1 / (1 + math.exp(exponent))


def player_interception_probability(
    player_pos: tuple,
    pass_start: tuple,
    pass_end: tuple,
    n_steps: int = N_STEPS,
) -> float:
    """
    Probabilidad máxima de intercepción de un jugador a lo largo de la trayectoria del pase.

    Se discretiza la trayectoria en n_steps puntos y se toma el máximo de probabilidad logística.
    """
    pass_dist = distance(pass_start, pass_end)
    if pass_dist == 0:
        return 0.0

    return max(
        interception_logistic_probability(
            distance(player_pos, interpolate(pass_start, pass_end, step / n_steps)),
            (step / n_steps) * pass_dist,
            pass_dist,
        )
        for step in range(1, n_steps + 1)
    )


def combined_probability(probabilities: list[float]) -> float:
    """Probabilidad combinada de al menos un evento independiente (regla del complemento)."""
    result = 1.0
    for p in probabilities:
        result *= (1 - p)
    return 1 - result


def pass_risk_metrics(
    shooter_pos: tuple,
    teammate_pos: tuple,
    opponents_df: pd.DataFrame,
    teammates_df: pd.DataFrame | None = None,
) -> dict:
    """
    Calcula las métricas de riesgo de un pase.

    - interception_probability: prob. de que un rival intercepte el pase
    - retention_probability: prob. de que un compañero retenga el balón
    - pass_risk: diferencia (mayor → más arriesgado)
    """
    opp_probs = [
        player_interception_probability(as_point(loc), shooter_pos, teammate_pos)
        for loc in opponents_df["location"]
    ]
    interception_prob = combined_probability(opp_probs)

    if teammates_df is None or teammates_df.empty:
        retention_prob = player_interception_probability(teammate_pos, shooter_pos, teammate_pos)
    else:
        available = teammates_df[
            teammates_df["location"].apply(
                lambda loc: distance(as_point(loc), shooter_pos) > 1e-6
            )
        ]
        tm_probs = [
            player_interception_probability(as_point(loc), shooter_pos, teammate_pos)
            for loc in available["location"]
        ]
        retention_prob = combined_probability(tm_probs) if tm_probs else \
            player_interception_probability(teammate_pos, shooter_pos, teammate_pos)

    return {
        "interception_probability": interception_prob,
        "retention_probability": retention_prob,
        "pass_risk": interception_prob - retention_prob,
    }


def is_pass_lane_clear(
    shooter_pos: tuple,
    teammate_pos: tuple,
    opponents_df: pd.DataFrame,
    teammates_df: pd.DataFrame | None = None,
) -> bool:
    """Devuelve True si el pase cumple los umbrales de intercepción y riesgo."""
    if opponents_df.empty:
        return True
    metrics = pass_risk_metrics(shooter_pos, teammate_pos, opponents_df, teammates_df)
    return (
        metrics["interception_probability"] < MAX_INTERCEPTION_PROB and
        metrics["pass_risk"] <= MAX_PASS_RISK
    )


def is_valid_teammate_position(
    shooter_pos: tuple,
    teammate_pos: tuple,
    opponents_df: pd.DataFrame,
) -> bool:
    """
    Comprueba si el compañero está en posición válida (no en fuera de juego).

    Regla simplificada: el compañero debe estar por delante del segundo último defensor.
    Se excluyen posiciones en campo propio o por detrás del balón.
    """
    teammate_x = teammate_pos[0]
    ball_x = shooter_pos[0]

    if teammate_x <= 60 or teammate_x <= ball_x:
        return True
    if len(opponents_df) < 2:
        return True

    sorted_defenders_x = sorted(
        (as_point(loc)[0] for loc in opponents_df["location"]),
        reverse=True,
    )
    return teammate_x <= sorted_defenders_x[1]


# =============================================================================
# ANÁLISIS DE UN TIRO
# =============================================================================

def analyze_shot(shot: pd.Series, goal_events: pd.DataFrame) -> dict | None:
    """
    Analiza la toma de decisión de un tiro individual.

    Devuelve un diccionario con todas las métricas o None si no hay compañeros visibles.
    """
    player_pos = tuple(shot["location"])
    frame = pd.DataFrame(shot["freeze_frame"])
    teammates = frame[frame["teammate"]].copy().reset_index(drop=True)
    opponents = frame[~frame["teammate"]].copy().reset_index(drop=True)

    if teammates.empty:
        return None

    # Métricas del tirador
    player_m = scoring_probability(player_pos, opponents)
    player_angle = shot_angle(player_pos)

    # Métricas de cada compañero
    tm_metrics = teammates["location"].apply(
        lambda loc: pd.Series(scoring_probability(as_point(loc), opponents))
    )
    teammates = pd.concat([teammates, tm_metrics], axis=1)
    teammates["valid_position"] = teammates["location"].apply(
        lambda loc: is_valid_teammate_position(player_pos, as_point(loc), opponents)
    )

    # Rankings
    valid_tm = teammates[teammates["valid_position"]]
    shooter_rank = 1 + int((teammates["p_score"] > player_m["p_score"]).sum())
    shooter_rank_valid = 1 + int((valid_tm["p_score"] > player_m["p_score"]).sum())

    # Compañeros con mejor p_score y posición válida
    better_tm = teammates[
        (teammates["p_score"] > player_m["p_score"]) & teammates["valid_position"]
    ].copy().sort_values("p_score", ascending=False)

    n_better_invalid = int(
        ((teammates["p_score"] > player_m["p_score"]) & ~teammates["valid_position"]).sum()
    )

    # Evaluación de líneas de pase
    if better_tm.empty:
        better_options = better_tm.copy()
    else:
        risk_data = better_tm["location"].apply(
            lambda loc: pd.Series(pass_risk_metrics(player_pos, as_point(loc), opponents, teammates))
        )
        better_tm = pd.concat([better_tm.reset_index(drop=True), risk_data.reset_index(drop=True)], axis=1)
        better_tm["pass_clear"] = better_tm["location"].apply(
            lambda loc: is_pass_lane_clear(player_pos, as_point(loc), opponents, teammates)
        )
        better_options = better_tm[better_tm["pass_clear"]].copy()

    # Mejor opción de pase (maximiza valor esperado: prob_éxito × p_score compañero)
    if better_options.empty:
        best_pass = {
            "interception_probability": None,
            "retention_probability": None,
            "pass_risk": None,
            "best_teammate_value": None,
            "best_teammate_pass_success_prob": None,
        }
    else:
        better_options = better_options.copy()
        better_options["pass_option_value"] = (
            better_options["retention_probability"] * better_options["p_score"]
        )
        best = better_options.sort_values("pass_option_value", ascending=False).iloc[0]
        best_pass = {
            "interception_probability": best["interception_probability"],
            "retention_probability": best["retention_probability"],
            "pass_risk": best["pass_risk"],
            "best_teammate_value": best["pass_option_value"],
            "best_teammate_pass_success_prob": best["retention_probability"],
        }

    shooter_value = player_m["p_score"]
    best_teammate_value = best_pass["best_teammate_value"]
    decision_value = (
        shooter_value - best_teammate_value
        if best_teammate_value is not None
        else shooter_value
    )
    decision_margin = (
        shooter_value - best_teammate_value
        if best_teammate_value is not None
        else 0.0
    )

    match_id = int(shot.get("match_id") or shot.get("match_id_x") or shot.get("match_id_y"))
    live_score_diff = score_difference_before_event(match_id, shot["event_index"], goal_events)

    return {
        "shot_id":                         shot["id"],
        "match_id":                        match_id,
        "player":                          shot["player.name"],
        "team":                            shot["team.name"],
        "period":                          shot["period"],
        "minute":                          shot["minute"],
        "second":                          shot["second"],
        "time_window":                     time_window(shot["period"], shot["minute"]),
        "score_diff":                      live_score_diff,
        "score_state":                     score_state(live_score_diff),
        "location":                        shot["location"],
        "shot_angle":                      player_angle,
        "d_0":                             player_m["d_0"],
        "n_def":                           player_m["n_def"],
        "d_m":                             player_m["d_m"],
        "p_score":                         player_m["p_score"],
        "shooter_rank":                    shooter_rank,
        "n_shooting_options":              len(teammates) + 1,
        "shooter_rank_pct":                shooter_rank / (len(teammates) + 1),
        "shooter_is_best_option":          shooter_rank == 1,
        "shooter_rank_valid":              shooter_rank_valid,
        "n_valid_shooting_options":        len(valid_tm) + 1,
        "shooter_rank_valid_pct":          shooter_rank_valid / (len(valid_tm) + 1),
        "shooter_is_best_valid_option":    shooter_rank_valid == 1,
        "best_teammate_p_score":           float(teammates["p_score"].max()),
        "best_valid_teammate_p_score":     float(valid_tm["p_score"].max()) if not valid_tm.empty else None,
        "best_option_p_score":             max(shooter_value, float(valid_tm["p_score"].max()) if not valid_tm.empty else float("-inf")),
        "n_better_invalid_position":       n_better_invalid,
        "n_better_teammates":              len(better_tm),
        "n_better_options":                len(better_options),
        "best_pass_interception_prob":     best_pass["interception_probability"],
        "best_pass_retention_prob":        best_pass["retention_probability"],
        "best_pass_risk":                  best_pass["pass_risk"],
        "shooter_value":                   shooter_value,
        "best_teammate_value":             best_teammate_value,
        "best_teammate_pass_success_prob": best_pass["best_teammate_pass_success_prob"],
        "decision_value":                  decision_value,
        "decision_margin":                 decision_margin,
        "good_decision":                   better_options.empty,
        "shot_outcome":                    shot.get("shot.outcome.name"),
        "shot_xg":                         shot.get("shot.statsbomb_xg"),
    }


# =============================================================================
# VISUALIZACIONES METODOLÓGICAS
# =============================================================================

def plot_shot_example(shot: pd.Series, output_dir: Path) -> None:
    """
    Genera y guarda la figura metodológica de un tiro de ejemplo:
    campo con tirador, compañeros, rivales y triángulo de tiro.
    """
    player_pos = tuple(shot["location"])
    frame = pd.DataFrame(shot["freeze_frame"])
    teammates = frame[frame["teammate"]].copy()
    opponents = frame[~frame["teammate"]].copy()

    pitch = Pitch(pitch_type="statsbomb")
    fig, ax = pitch.draw(figsize=(10, 7))

    pitch.scatter(player_pos[0], player_pos[1], ax=ax, color="black", s=120, label="Tirador", zorder=5)
    if not teammates.empty:
        pitch.scatter(
            teammates["location"].apply(lambda x: x[0]),
            teammates["location"].apply(lambda x: x[1]),
            ax=ax, color="blue", s=80, label="Compañeros",
        )
    if not opponents.empty:
        pitch.scatter(
            opponents["location"].apply(lambda x: x[0]),
            opponents["location"].apply(lambda x: x[1]),
            ax=ax, color="red", s=80, label="Rivales",
        )

    pitch.lines(player_pos[0], player_pos[1], LEFT_POST[0], LEFT_POST[1], ax=ax, color="orange", lw=2)
    pitch.lines(player_pos[0], player_pos[1], RIGHT_POST[0], RIGHT_POST[1], ax=ax, color="orange", lw=2)

    ax.legend()
    ax.set_title("Metodología: triángulo de tiro y jugadores en campo")
    fig.tight_layout()
    fig.savefig(output_dir / "metodologia_triangulo_tiro.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_pass_example(shot: pd.Series, best_option_pos: tuple, output_dir: Path) -> None:
    """
    Genera y guarda la figura metodológica de la mejor opción de pase detectada
    para un tiro de ejemplo.
    """
    player_pos = tuple(shot["location"])
    frame = pd.DataFrame(shot["freeze_frame"])
    teammates = frame[frame["teammate"]].copy()
    opponents = frame[~frame["teammate"]].copy()

    pitch = Pitch(pitch_type="statsbomb")
    fig, ax = pitch.draw(figsize=(10, 7))

    pitch.scatter(player_pos[0], player_pos[1], ax=ax, color="black", s=120, label="Tirador", zorder=5)
    if not teammates.empty:
        pitch.scatter(
            teammates["location"].apply(lambda x: x[0]),
            teammates["location"].apply(lambda x: x[1]),
            ax=ax, color="blue", s=80, label="Compañeros",
        )
    if not opponents.empty:
        pitch.scatter(
            opponents["location"].apply(lambda x: x[0]),
            opponents["location"].apply(lambda x: x[1]),
            ax=ax, color="red", s=80, label="Rivales",
        )

    pitch.scatter(best_option_pos[0], best_option_pos[1], ax=ax, color="yellow", s=180, label="Mejor opción", zorder=6)
    pitch.lines(player_pos[0], player_pos[1], LEFT_POST[0], LEFT_POST[1], ax=ax, color="orange", lw=2)
    pitch.lines(player_pos[0], player_pos[1], RIGHT_POST[0], RIGHT_POST[1], ax=ax, color="orange", lw=2)
    pitch.lines(player_pos[0], player_pos[1], best_option_pos[0], best_option_pos[1], ax=ax, color="green", lw=3)

    ax.legend()
    ax.set_title("Metodología: mejor opción de pase detectada")
    fig.tight_layout()
    fig.savefig(output_dir / "metodologia_mejor_pase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_best_pass_example(shots_360: pd.DataFrame) -> tuple[pd.Series | None, tuple | None]:
    """
    Busca el primer tiro del dataset que tenga al menos una mejor opción de pase clara,
    para usarlo como figura metodológica representativa.
    """
    for _, shot in shots_360.iterrows():
        player_pos = tuple(shot["location"])
        frame = pd.DataFrame(shot["freeze_frame"])
        teammates = frame[frame["teammate"]].copy().reset_index(drop=True)
        opponents = frame[~frame["teammate"]].copy().reset_index(drop=True)

        if teammates.empty:
            continue

        player_m = scoring_probability(player_pos, opponents)
        tm_metrics = teammates["location"].apply(
            lambda loc: pd.Series(scoring_probability(as_point(loc), opponents))
        )
        teammates = pd.concat([teammates, tm_metrics], axis=1)
        teammates["valid_position"] = teammates["location"].apply(
            lambda loc: is_valid_teammate_position(player_pos, as_point(loc), opponents)
        )

        better_tm = teammates[
            (teammates["p_score"] > player_m["p_score"]) & teammates["valid_position"]
        ].copy().reset_index(drop=True)

        if better_tm.empty:
            continue

        risk_data = better_tm["location"].apply(
            lambda loc: pd.Series(pass_risk_metrics(player_pos, as_point(loc), opponents, teammates))
        )
        better_tm = pd.concat([better_tm, risk_data.reset_index(drop=True)], axis=1)
        better_tm["pass_clear"] = better_tm["location"].apply(
            lambda loc: is_pass_lane_clear(player_pos, as_point(loc), opponents, teammates)
        )
        better_options = better_tm[better_tm["pass_clear"]].copy()

        if not better_options.empty:
            better_options["pass_option_value"] = (
                better_options["retention_probability"] * better_options["p_score"]
            )
            best_pos = tuple(better_options.sort_values("pass_option_value", ascending=False).iloc[0]["location"])
            return shot, best_pos

    return None, None


def plot_ranking_with_field(shots_360: pd.DataFrame, output_dir: Path) -> None:
    """
    Figura de resultado: ejemplo de ranking del tirador respecto a compañeros.
    Busca un tiro donde el tirador no sea la mejor opción y colorea los jugadores
    por p_score (verde=mejor, rojo=peor). Incluye el triángulo de tiro.
    """
    for _, shot in shots_360.iterrows():
        player_pos = tuple(shot["location"])
        frame = pd.DataFrame(shot["freeze_frame"])
        teammates = frame[frame["teammate"]].copy().reset_index(drop=True)
        opponents = frame[~frame["teammate"]].copy().reset_index(drop=True)

        if len(teammates) < 3:
            continue

        player_m = scoring_probability(player_pos, opponents)
        tm_metrics = teammates["location"].apply(
            lambda loc: pd.Series(scoring_probability(as_point(loc), opponents))
        )
        teammates = pd.concat([teammates, tm_metrics], axis=1)

        better_count = int((teammates["p_score"] > player_m["p_score"]).sum())
        if better_count < 2:
            continue

        # Tenemos un buen ejemplo — dibujar
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a472a", line_color="white",
                      goal_type="box")
        fig, ax = pitch.draw(figsize=(11, 7))  # type: ignore[misc]

        all_ps = list(teammates["p_score"]) + [player_m["p_score"]]
        pmin, pmax = min(all_ps), max(all_ps)
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

        # Rivales
        if not opponents.empty:
            pitch.scatter(  # type: ignore[misc]
                opponents["location"].apply(lambda x: x[0]),
                opponents["location"].apply(lambda x: x[1]),
                ax=ax, color="#B71C1C", s=100, edgecolors="white", lw=1, zorder=4,
            )

        # Compañeros coloreados por p_score
        for rank, (_, row) in enumerate(
            teammates.sort_values("p_score", ascending=False).iterrows(), start=1
        ):
            loc = as_point(row["location"])
            norm_p = (row["p_score"] - pmin) / (pmax - pmin + 1e-9)
            col = cmap(norm_p)
            pitch.scatter(loc[0], loc[1], ax=ax, color=col, s=180,  # type: ignore[misc]
                          edgecolors="white", lw=1.5, zorder=5)
            ax.text(loc[0], loc[1] - 3.5, f"#{rank}",  # type: ignore[misc]
                    ha="center", va="top", fontsize=7, color="white",
                    fontweight="bold", zorder=7)

        # Tirador
        rank_shooter = int((teammates["p_score"] > player_m["p_score"]).sum()) + 1
        norm_s = (player_m["p_score"] - pmin) / (pmax - pmin + 1e-9)
        pitch.scatter(player_pos[0], player_pos[1], ax=ax,  # type: ignore[misc]
                      color=cmap(norm_s), s=300, edgecolors="white", lw=2.5,
                      zorder=6, marker="*")
        ax.text(player_pos[0], player_pos[1] - 4,  # type: ignore[misc]
                f"Tirador\n#{rank_shooter}", ha="center", va="top",
                fontsize=7, color="white", fontweight="bold", zorder=8)

        # Triángulo
        tri_x = [player_pos[0], LEFT_POST[0], RIGHT_POST[0], player_pos[0]]
        tri_y = [player_pos[1], LEFT_POST[1], RIGHT_POST[1], player_pos[1]]
        ax.plot(tri_x, tri_y, color="#E65100", lw=1.5, linestyle="--",  # type: ignore[misc]
                alpha=0.7, zorder=3)

        player_name = str(shot.get("player.name", ""))
        ax.set_title(  # type: ignore[misc]
            f"Ranking de opciones de tiro — {player_name.split()[-1]} "
            f"(min. {int(shot['minute'])})\n"
            f"Tirador ocupa la posición #{rank_shooter} de {len(teammates) + 1} | "
            f"p(marcar)={player_m['p_score']:.4f} | color: verde=mayor p_score",
            fontsize=9, color="white", pad=8,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap,  # type: ignore[attr-defined]
                                   norm=plt.Normalize(pmin, pmax))  # type: ignore[attr-defined]
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("p(marcar)", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        fig.savefig(output_dir / "ranking_compañeros_ejemplo.png",
                    dpi=150, bbox_inches="tight", facecolor="#1a472a")
        plt.close(fig)
        return



_C_GOOD = "#2E7D32"
_C_BAD  = "#C62828"
_C_MID  = "#EF6C00"


def plot_decision_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Distribución buenas/malas decisiones: barras + gráfico de pastel."""
    counts = df["good_decision"].value_counts()
    n_good = counts.get(True, 0)
    n_bad  = counts.get(False, 0)
    total  = n_good + n_bad

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Barras
    ax = axes[0]
    bars = ax.bar(["Buena decisión", "Mala decisión"], [n_good, n_bad],
                  color=[_C_GOOD, _C_BAD], edgecolor="white", width=0.5)
    for bar, n in zip(bars, [n_good, n_bad]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{n}\n({n / total * 100:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(n_good, n_bad) * 1.18)
    ax.set_ylabel("Número de tiros", fontsize=10)
    ax.set_title("Distribución de decisiones", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Pastel
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        [n_good, n_bad],
        labels=["Buena\ndecisión", "Mala\ndecisión"],
        colors=[_C_GOOD, _C_BAD],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.set_title(f"Total: {total} tiros analizados", fontsize=11)

    fig.suptitle("Calidad de la toma de decisión en el tiro — Bayer Leverkusen 2023/24",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "decision_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_xg_by_decision(df: pd.DataFrame, output_dir: Path) -> None:
    """xG medio según calidad de decisión, con anotación de diferencia."""
    xg_mean = df.groupby("good_decision")["shot_xg"].mean()
    xg_bad  = xg_mean.get(False, 0.0)
    xg_good = xg_mean.get(True,  0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Mala decisión", "Buena decisión"], [xg_bad, xg_good],
                  color=[_C_BAD, _C_GOOD], edgecolor="white", width=0.45)
    for bar, val in zip(bars, [xg_bad, xg_good]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("xG medio", fontsize=10)
    ax.set_title("xG medio según calidad de decisión", fontsize=11)
    ax.set_ylim(0, max(xg_bad, xg_good) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "xg_by_decision.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_xg_vs_alternatives(df: pd.DataFrame, output_dir: Path) -> None:
    """xG vs número de alternativas disponibles, con media por grupo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    good = df[df["good_decision"]]
    bad  = df[~df["good_decision"]]

    ax.scatter(good["n_better_options"], good["shot_xg"],
               color=_C_GOOD, alpha=0.4, s=30, label="Buena decisión", edgecolors="none")
    ax.scatter(bad["n_better_options"],  bad["shot_xg"],
               color=_C_BAD,  alpha=0.4, s=30, label="Mala decisión",  edgecolors="none")

    for subset, color in [(good, _C_GOOD), (bad, _C_BAD)]:
        grp = subset.groupby("n_better_options")["shot_xg"].mean()
        ax.plot(grp.index, grp.values, "o-", color=color, lw=2, ms=6, zorder=5)

    ax.set_xlabel("Número de mejores opciones disponibles", fontsize=10)
    ax.set_ylabel("xG del tiro", fontsize=10)
    ax.set_title("xG del tiro según alternativas de pase disponibles", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "xg_vs_alternatives.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_player_ranking(df: pd.DataFrame, output_dir: Path) -> None:
    """Ranking horizontal de jugadores con colores por umbral y etiqueta de n_shots."""
    player_summary = (
        df.groupby("player")
        .agg(good_decision=("good_decision", "mean"), n_shots=("shot_id", "count"))
        .query("n_shots >= 5")
        .sort_values("good_decision", ascending=True)
        .reset_index()
    )

    colors = [
        _C_GOOD if v >= 0.55 else (_C_BAD if v < 0.45 else _C_MID)
        for v in player_summary["good_decision"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(5, len(player_summary) * 0.42)))
    bars = ax.barh(player_summary["player"], player_summary["good_decision"] * 100,
                   color=colors, edgecolor="white", height=0.65)

    for bar, n in zip(bars, player_summary["n_shots"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=8, color="#555555")

    ax.axvline(50, color="#999999", lw=1, linestyle="--")
    ax.set_xlabel("% buenas decisiones", fontsize=10)
    ax.set_title("Ranking de jugadores según toma de decisión en el tiro\n"
                 "(jugadores con ≥5 tiros)", fontsize=11)
    ax.set_xlim(0, 105)

    import matplotlib.patches as mpatches
    legend_el = [
        mpatches.Patch(color=_C_GOOD, label=">55% buenas"),
        mpatches.Patch(color=_C_MID,  label="45–55%"),
        mpatches.Patch(color=_C_BAD,  label="<45% buenas"),
    ]
    ax.legend(handles=legend_el, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "player_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decision_by_outcome(df: pd.DataFrame, output_dir: Path) -> None:
    """Proporción de buenas decisiones según resultado del tiro."""
    outcome_data = df.groupby("shot_outcome")["good_decision"].mean().sort_values()
    colors = [_C_GOOD if v >= 0.55 else (_C_BAD if v < 0.45 else _C_MID)
              for v in outcome_data.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(outcome_data)), outcome_data.values * 100,
                  color=colors, edgecolor="white")
    for bar, val in zip(bars, outcome_data.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(outcome_data)))
    ax.set_xticklabels(outcome_data.index, rotation=35, ha="right", fontsize=9)
    ax.axhline(50, color="#999999", lw=1, linestyle="--")
    ax.set_ylabel("% buenas decisiones", fontsize=10)
    ax.set_title("Proporción de buenas decisiones según resultado del tiro", fontsize=11)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "decision_by_outcome.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decision_heatmap(
    df: pd.DataFrame,
    window_col: str,
    window_order: list[str],
    score_col: str,
    score_order: list[str],
    output_stem: str,
    title: str,
    output_dir: Path,
    value_col: str = "decision_value",
    colorbar_label: str = "decision_value medio",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera y guarda un heatmap de una métrica media por ventana temporal y estado del marcador.

    Devuelve (value_matrix, count_matrix).
    """
    data = df.copy()
    data[window_col] = pd.Categorical(data[window_col], categories=window_order, ordered=True)
    data[score_col] = pd.Categorical(data[score_col], categories=score_order, ordered=True)

    value_matrix: pd.DataFrame = data.pivot_table(
        index=window_col, columns=score_col,
        values=value_col, aggfunc="mean", observed=False,
    ).reindex(index=window_order, columns=score_order)

    count_matrix: pd.DataFrame = data.pivot_table(
        index=window_col, columns=score_col,
        values="shot_id", aggfunc="count", fill_value=0, observed=False,
    ).reindex(index=window_order, columns=score_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    heatmap = ax.imshow(value_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(score_order)))
    ax.set_xticklabels(score_order)
    ax.set_yticks(range(len(window_order)))
    ax.set_yticklabels(window_order)
    ax.set_xlabel("Estado del marcador (perspectiva Bayer)")
    ax.set_ylabel("Ventana temporal")
    ax.set_title(title)

    for r, window in enumerate(window_order):
        for c, score in enumerate(score_order):
            n = int(count_matrix.loc[window, score])
            val = value_matrix.loc[window, score]
            label = "n=0" if n == 0 else f"{val:.3f}\nn={n}"
            ax.text(c, r, label, ha="center", va="center", fontsize=9)

    fig.colorbar(heatmap, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_dir / f"heatmap_{output_stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    value_matrix.to_csv(output_dir / f"matrix_{value_col}_{output_stem}.csv", encoding="utf-8-sig")
    count_matrix.to_csv(output_dir / f"matrix_shot_count_{output_stem}.csv", encoding="utf-8-sig")

    return value_matrix, count_matrix


# =============================================================================
# GUARDADO DE TABLAS
# =============================================================================

def save_result_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Guarda todas las tablas de resultados en CSV."""
    df.to_csv(output_dir / "decision_results.csv", index=False, encoding="utf-8-sig")

    player_summary = (
        df.groupby("player")
        .agg(good_decision=("good_decision", "mean"), n_shots=("shot_id", "count"))
        .sort_values("good_decision", ascending=False)
        .reset_index()
    )
    player_summary.to_csv(output_dir / "player_decision_summary.csv", index=False, encoding="utf-8-sig")

    df["player"].value_counts().rename_axis("player").reset_index(name="n_shots") \
        .to_csv(output_dir / "player_shot_counts.csv", index=False, encoding="utf-8-sig")

    df.groupby("shot_outcome")["good_decision"].mean().sort_values(ascending=False) \
        .reset_index().to_csv(output_dir / "decision_by_outcome.csv", index=False, encoding="utf-8-sig")

    df.groupby("good_decision")["shot_xg"].mean().reset_index() \
        .to_csv(output_dir / "xg_by_decision.csv", index=False, encoding="utf-8-sig")

    df[~df["good_decision"]].to_csv(output_dir / "bad_decision_examples.csv", index=False, encoding="utf-8-sig")

    df.groupby("good_decision")[["shooter_value", "best_teammate_value", "decision_value", "decision_margin"]].mean() \
        .reset_index().to_csv(output_dir / "decision_value_summary.csv", index=False, encoding="utf-8-sig")

    good_decision_rate = df["good_decision"].mean()
    pd.DataFrame([{
        "shots_analyzed":           len(df),
        "good_decision_rate":       good_decision_rate,
        "good_decision_percentage": good_decision_rate * 100,
        "bad_decisions":            (~df["good_decision"]).sum(),
        "mean_decision_value":      df["decision_value"].mean(),
        "total_decision_value":     df["decision_value"].sum(),
        "mean_decision_margin":     df["decision_margin"].mean(),
        "total_decision_margin":    df["decision_margin"].sum(),
    }]).to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # --- Carga de datos ---
    matches = load_matches(DATA_DIR)
    valid_match_ids = get_valid_match_ids(matches)
    print(f"Partidos con datos 360 disponibles: {len(valid_match_ids)}")

    df_events, df_360 = load_all_events_and_360(valid_match_ids)
    print(f"Eventos totales: {df_events.shape[0]:,} | Registros 360: {df_360.shape[0]:,}")

    goal_events = build_goal_events(df_events)

    # --- Tiros del Bayer con datos 360 ---
    shots_bayer = df_events[
        (df_events["type.name"] == "Shot") &
        (df_events["team.name"] == "Bayer Leverkusen")
    ].copy()

    shots_360 = shots_bayer.merge(df_360, left_on="id", right_on="event_uuid", how="inner")
    print(f"Tiros del Bayer: {len(shots_bayer)} | Con datos 360: {len(shots_360)}")

    # --- Figuras metodológicas (guardadas antes del análisis completo) ---
    first_shot = shots_360.iloc[0]
    plot_shot_example(first_shot, OUTPUT_DIR)

    example_shot, best_pos = find_best_pass_example(shots_360)
    if example_shot is not None and best_pos is not None:
        plot_best_pass_example(example_shot, best_pos, OUTPUT_DIR)

    # --- Análisis de todos los tiros ---
    results = []
    for _, shot in shots_360.iterrows():
        result = analyze_shot(shot, goal_events)
        if result is not None:
            results.append(result)

    df_results = pd.DataFrame(results)
    print(f"\nTiros analizados: {len(df_results)}")

    # --- Métricas globales ---
    good_rate = df_results["good_decision"].mean()
    bad_count = (~df_results["good_decision"]).sum()
    print(f"Buenas decisiones: {good_rate * 100:.2f}%")
    print(f"Malas decisiones:  {bad_count}")

    # --- Guardado de tablas ---
    save_result_tables(df_results, OUTPUT_DIR)

    # --- Heatmap (4 ventanas × 5 estados de marcador) ---
    plot_decision_heatmap(
        df_results,
        window_col="time_window",
        window_order=TIME_WINDOW_ORDER,
        score_col="score_state",
        score_order=SCORE_STATE_ORDER,
        output_stem="4ventanas_5marcadores",
        title="Decision value medio por ventana temporal y estado del marcador",
        output_dir=OUTPUT_DIR,
    )
    plot_decision_heatmap(
        df_results,
        window_col="time_window",
        window_order=TIME_WINDOW_ORDER,
        score_col="score_state",
        score_order=SCORE_STATE_ORDER,
        output_stem="decision_margin_4ventanas_5marcadores",
        title="Decision margin medio por ventana temporal y estado del marcador",
        output_dir=OUTPUT_DIR,
        value_col="decision_margin",
        colorbar_label="decision_margin medio",
    )

    # --- Gráficos de resultados ---
    plot_decision_distribution(df_results, OUTPUT_DIR)
    plot_xg_by_decision(df_results, OUTPUT_DIR)
    plot_xg_vs_alternatives(df_results, OUTPUT_DIR)
    plot_player_ranking(df_results, OUTPUT_DIR)
    plot_decision_by_outcome(df_results, OUTPUT_DIR)
    plot_ranking_with_field(shots_360, OUTPUT_DIR)

    print(f"\nOutputs guardados en: {OUTPUT_DIR}")

    # --- Red de pases previa al tiro ---
    run_network_analysis(shots_360, df_events, df_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
