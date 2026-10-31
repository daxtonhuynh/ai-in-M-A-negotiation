from __future__ import annotations

import csv
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from merger_negotiation_game import (
    Action,
    BilateralMergerNegotiation,
    NegotiationState,
    alpha_beta,
)


# ------------------------------------------------------------
# Experiment configuration containers
# ------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    scenario_class: str
    buyer_reservation: float
    seller_reservation: float
    synergy: float
    delay_cost: float
    buyer_concession_budget: float
    seller_concession_budget: float
    max_rounds: int
    offer_step: float = 5.0
    breakdown_penalty: float = 20.0


@dataclass(frozen=True)
class MethodConfig:
    method_label: str
    method: str
    display_name: str
    method_family: str
    ordering_variant: str = "n/a"
    use_move_ordering: bool = True
    depth_limit: Optional[int] = None
    heuristic: Optional[str] = None
    include_in_main_plots: bool = True


# ------------------------------------------------------------
# Action helpers
# ------------------------------------------------------------
def action_type(action: Optional[Action]) -> str:
    return action[0] if action is not None else "none"


def action_price(action: Optional[Action]) -> float:
    if action is None or action[1] is None:
        return float("nan")
    return float(action[1])


def serialize_action(action: Optional[Action]) -> str:
    if action is None:
        return "none"
    kind, price = action
    if price is None:
        return kind
    if float(price).is_integer():
        return f"{kind}@{int(price)}"
    return f"{kind}@{price:.2f}"


def actions_equal(a1: Optional[Action], a2: Optional[Action], tol: float = 1e-9) -> bool:
    if a1 is None or a2 is None:
        return a1 is None and a2 is None
    if a1[0] != a2[0]:
        return False
    p1, p2 = a1[1], a2[1]
    if p1 is None or p2 is None:
        return p1 is None and p2 is None
    return abs(float(p1) - float(p2)) <= tol


# ------------------------------------------------------------
# Counted search wrappers
# ------------------------------------------------------------
def minimax_counted(
    game: BilateralMergerNegotiation,
    state: NegotiationState,
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[float, Optional[Action]]:
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1

    if game.is_terminal(state):
        return game.utility(state), None

    if game.to_move(state) == "buyer":
        best_value = -math.inf
        best_action: Optional[Action] = None
        for action in game.actions(state):
            value, _ = minimax_counted(game, game.result(state, action), stats)
            if value > best_value:
                best_value, best_action = value, action
        return best_value, best_action

    best_value = math.inf
    best_action = None
    for action in game.actions(state):
        value, _ = minimax_counted(game, game.result(state, action), stats)
        if value < best_value:
            best_value, best_action = value, action
    return best_value, best_action



def solve_one_step_counted(
    game: BilateralMergerNegotiation,
    state: NegotiationState,
    method_config: MethodConfig,
) -> Tuple[float, Optional[Action], Dict[str, float]]:
    start = time.perf_counter()
    stats: Dict[str, float] = {"nodes": 0, "cutoff_events": 0, "pruned_siblings": 0}

    if method_config.method == "minimax":
        value, action = minimax_counted(game, state, stats)
    elif method_config.method == "alphabeta":
        value, action = alpha_beta(game, state, depth_limit=None, eval_fn=None, stats=stats)
    elif method_config.method == "heuristic_alphabeta":
        if not method_config.heuristic:
            raise ValueError("heuristic_alphabeta requires a heuristic name.")
        eval_fn = getattr(game, method_config.heuristic)
        value, action = alpha_beta(
            game,
            state,
            depth_limit=method_config.depth_limit,
            eval_fn=eval_fn,
            stats=stats,
        )
    else:
        raise ValueError(f"Unsupported method: {method_config.method}")

    stats["runtime_sec"] = time.perf_counter() - start
    return value, action, stats



def play_negotiation_counted(
    game: BilateralMergerNegotiation,
    method_config: MethodConfig,
) -> Dict[str, float | str]:
    """
    Play out one full negotiation by re-solving from each decision state.

    Reported node/runtime totals are cumulative across all decision episodes until the
    negotiation reaches a terminal outcome.
    """
    state = game.initial_state
    totals: Dict[str, float | str] = {
        "nodes": 0.0,
        "cutoff_events": 0.0,
        "pruned_siblings": 0.0,
        "runtime_sec": 0.0,
        "decision_steps": 0.0,
        "root_action_type": "none",
        "root_action_price": float("nan"),
        "root_action_label": "none",
        "root_value": float("nan"),
    }

    first_decision = True

    while not game.is_terminal(state):
        value, action, step_stats = solve_one_step_counted(game, state, method_config)
        totals["nodes"] = float(totals["nodes"]) + step_stats.get("nodes", 0)
        totals["cutoff_events"] = float(totals["cutoff_events"]) + step_stats.get("cutoff_events", 0)
        totals["pruned_siblings"] = float(totals["pruned_siblings"]) + step_stats.get("pruned_siblings", 0)
        totals["runtime_sec"] = float(totals["runtime_sec"]) + step_stats.get("runtime_sec", 0.0)
        totals["decision_steps"] = float(totals["decision_steps"]) + 1
        totals["last_root_value"] = value

        if first_decision:
            totals["root_action_type"] = action_type(action)
            totals["root_action_price"] = action_price(action)
            totals["root_action_label"] = serialize_action(action)
            totals["root_value"] = value
            first_decision = False

        if action is None:
            from dataclasses import replace

            state = replace(state, status="horizon")
            break

        state = game.result(state, action)

        if state.round_no > state.max_rounds and state.status == "ongoing":
            from dataclasses import replace

            state = replace(state, status="horizon")

    terminal_round = min(state.round_no, state.max_rounds)
    totals["terminal_round"] = float(terminal_round)
    totals["rounds"] = float(terminal_round)
    totals["deal_success"] = 1.0 if state.status == "accepted" else 0.0
    totals["final_price"] = state.agreed_price if state.agreed_price is not None else float("nan")
    totals["utility"] = game.utility(state)
    totals["final_status"] = state.status
    totals["agreement_round"] = float(state.round_no) if state.status == "accepted" else float("nan")
    totals["cutoff_rate"] = (
        float(totals["cutoff_events"]) / float(totals["nodes"]) if float(totals["nodes"]) else 0.0
    )
    return totals


# ------------------------------------------------------------
# Scenario and method grids
# ------------------------------------------------------------
def build_scenarios() -> List[ScenarioConfig]:
    """
    Controlled, reproducible synthetic scenarios grouped into three bargaining regimes.

    - buyer_advantaged: wider feasible zone, lower delay cost, stronger buyer flexibility
    - balanced: moderate feasible zone and balanced concession budgets
    - seller_advantaged: tighter zone, higher delay cost, stronger seller flexibility
    """
    return [
        ScenarioConfig("BA1", "buyer_advantaged", 100, 70, 10, 0, 20, 30, 3),
        ScenarioConfig("BA2", "buyer_advantaged", 100, 80, 20, 0, 20, 30, 3),
        ScenarioConfig("BA3", "buyer_advantaged", 110, 85, 15, 0, 20, 30, 3),
        ScenarioConfig("BL1", "balanced", 105, 85, 10, 0, 20, 20, 3),
        ScenarioConfig("BL2", "balanced", 110, 90, 10, 0, 20, 20, 3),
        ScenarioConfig("BL3", "balanced", 115, 95, 10, 0, 20, 20, 3),
        ScenarioConfig("SA1", "seller_advantaged", 110, 90, 5, 3, 15, 25, 3),
        ScenarioConfig("SA2", "seller_advantaged", 115, 95, 5, 3, 15, 25, 3),
        ScenarioConfig("SA3", "seller_advantaged", 120, 100, 5, 4, 15, 25, 3),
    ]



def build_methods(include_rq3_variants: bool = True) -> List[MethodConfig]:
    methods = [
        MethodConfig("minimax", "minimax", "Minimax", "minimax"),
        MethodConfig("alpha_beta", "alphabeta", "Alpha-Beta", "alpha_beta", ordering_variant="ordered"),
        MethodConfig(
            "hab_h1_d2",
            "heuristic_alphabeta",
            "HAB h1 (d=2)",
            "hab_h1_d2",
            ordering_variant="ordered",
            depth_limit=2,
            heuristic="h1",
        ),
        MethodConfig(
            "hab_h2_d2",
            "heuristic_alphabeta",
            "HAB h2 (d=2)",
            "hab_h2_d2",
            ordering_variant="ordered",
            depth_limit=2,
            heuristic="h2",
        ),
        MethodConfig(
            "hab_h3_d2",
            "heuristic_alphabeta",
            "HAB h3 (d=2)",
            "hab_h3_d2",
            ordering_variant="ordered",
            depth_limit=2,
            heuristic="h3",
        ),
        MethodConfig(
            "hab_h3_d3",
            "heuristic_alphabeta",
            "HAB h3 (d=3)",
            "hab_h3_d3",
            ordering_variant="ordered",
            depth_limit=3,
            heuristic="h3",
        ),
    ]

    if include_rq3_variants:
        methods.extend(
            [
                MethodConfig(
                    "alpha_beta_unordered",
                    "alphabeta",
                    "Alpha-Beta (unordered)",
                    "alpha_beta",
                    ordering_variant="unordered",
                    use_move_ordering=False,
                    include_in_main_plots=False,
                ),
                MethodConfig(
                    "hab_h3_d3_unordered",
                    "heuristic_alphabeta",
                    "HAB h3 (d=3, unordered)",
                    "hab_h3_d3",
                    ordering_variant="unordered",
                    use_move_ordering=False,
                    depth_limit=3,
                    heuristic="h3",
                    include_in_main_plots=False,
                ),
            ]
        )

    # Matched-depth comparisons isolate heuristic design from search depth.
    for heuristic in ("h1", "h2"):
        methods.append(MethodConfig(
            f"hab_{heuristic}_d3", "heuristic_alphabeta", f"HAB {heuristic} (d=3)",
            f"hab_{heuristic}_d3", "ordered", True, 3, heuristic, False))
    return methods


# ------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------
def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


NUMERIC_METRICS = [
    "runtime_sec",
    "nodes",
    "cutoff_events",
    "pruned_siblings",
    "cutoff_rate",
    "deal_success",
    "final_price",
    "utility",
    "rounds",
    "decision_steps",
    "terminal_round",
    "agreement_round",
    "root_value",
    "match_minimax_root_action",
    "match_minimax_root_action_type",
    "same_final_status_as_minimax",
    "utility_gap_vs_minimax",
    "absolute_utility_gap_vs_minimax",
    "price_gap_vs_minimax",
    "absolute_price_gap_vs_minimax",
    "root_value_gap_vs_minimax",
    "absolute_root_value_gap_vs_minimax",
    "node_ratio_vs_minimax",
    "node_reduction_vs_minimax",
    "runtime_ratio_vs_minimax",
    "runtime_reduction_vs_minimax",
]


def summarize_rows(rows: Sequence[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped[key].append(row)

    summary_rows: List[Dict[str, object]] = []

    for key, group in sorted(grouped.items()):
        summary: Dict[str, object] = {group_keys[i]: key[i] for i in range(len(group_keys))}
        summary["n_runs"] = len(group)

        for metric in NUMERIC_METRICS:
            vals = [float(row[metric]) for row in group if metric in row and not math.isnan(float(row[metric]))]
            summary[f"mean_{metric}"] = round(mean(vals), 6) if vals else float("nan")

        status_counts: Dict[str, int] = defaultdict(int)
        root_action_counts: Dict[str, int] = defaultdict(int)
        for row in group:
            status_counts[str(row["final_status"])] += 1
            root_action_counts[str(row["root_action_type"])] += 1

        summary["modal_final_status"] = max(status_counts.items(), key=lambda x: x[1])[0]
        summary["modal_root_action_type"] = max(root_action_counts.items(), key=lambda x: x[1])[0]
        summary_rows.append(summary)

    return summary_rows



def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Scenario baseline helper
# ------------------------------------------------------------
def build_game_from_scenario(
    scenario: ScenarioConfig,
    use_move_ordering: bool = True,
    exit_payoff_mode: str = "asymmetric_relative",
) -> BilateralMergerNegotiation:
    return BilateralMergerNegotiation(
        buyer_reservation=scenario.buyer_reservation,
        seller_reservation=scenario.seller_reservation,
        synergy=scenario.synergy,
        delay_cost=scenario.delay_cost,
        buyer_concession_budget=scenario.buyer_concession_budget,
        seller_concession_budget=scenario.seller_concession_budget,
        max_rounds=scenario.max_rounds,
        offer_step=scenario.offer_step,
        breakdown_penalty=scenario.breakdown_penalty,
        use_move_ordering=use_move_ordering,
        exit_payoff_mode=exit_payoff_mode,
    )


# ------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------
def run_batch_experiments(
    scenarios: Optional[Iterable[ScenarioConfig]] = None,
    methods: Optional[Iterable[MethodConfig]] = None,
    exit_payoff_mode: str = "asymmetric_relative",
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    scenarios = list(scenarios) if scenarios is not None else build_scenarios()
    methods = list(methods) if methods is not None else build_methods()

    minimax_method = next((method for method in methods if method.method_label == "minimax"), None)
    if minimax_method is None:
        raise ValueError("A minimax baseline method is required for vs-minimax comparisons.")

    raw_rows: List[Dict[str, object]] = []

    for scenario in scenarios:
        baseline_game = build_game_from_scenario(
            scenario,
            use_move_ordering=True,
            exit_payoff_mode=exit_payoff_mode,
        )
        baseline_metrics = play_negotiation_counted(baseline_game, minimax_method)
        baseline_root_action: Optional[Action] = None
        if str(baseline_metrics["root_action_type"]) != "none":
            baseline_root_action = (
                str(baseline_metrics["root_action_type"]),
                None if math.isnan(float(baseline_metrics["root_action_price"])) else float(baseline_metrics["root_action_price"]),
            )

        for method in methods:
            if method.method_label == minimax_method.method_label:
                metrics = dict(baseline_metrics)
            else:
                game = build_game_from_scenario(
                    scenario,
                    use_move_ordering=method.use_move_ordering,
                    exit_payoff_mode=exit_payoff_mode,
                )
                metrics = play_negotiation_counted(game, method)

            method_root_action: Optional[Action] = None
            if str(metrics["root_action_type"]) != "none":
                method_root_action = (
                    str(metrics["root_action_type"]),
                    None if math.isnan(float(metrics["root_action_price"])) else float(metrics["root_action_price"]),
                )

            baseline_final_price = float(baseline_metrics["final_price"])
            method_final_price = float(metrics["final_price"])
            price_gap = (
                method_final_price - baseline_final_price
                if not math.isnan(method_final_price) and not math.isnan(baseline_final_price)
                else float("nan")
            )

            baseline_nodes = float(baseline_metrics["nodes"])
            method_nodes = float(metrics["nodes"])
            baseline_runtime = float(baseline_metrics["runtime_sec"])
            method_runtime = float(metrics["runtime_sec"])
            baseline_root_value = float(baseline_metrics["root_value"])
            method_root_value = float(metrics["root_value"])

            row: Dict[str, object] = {
                "scenario_id": scenario.scenario_id,
                "scenario_class": scenario.scenario_class,
                "buyer_reservation": scenario.buyer_reservation,
                "seller_reservation": scenario.seller_reservation,
                "synergy": scenario.synergy,
                "delay_cost": scenario.delay_cost,
                "buyer_concession_budget": scenario.buyer_concession_budget,
                "seller_concession_budget": scenario.seller_concession_budget,
                "max_rounds": scenario.max_rounds,
                "offer_step": scenario.offer_step,
                "breakdown_penalty": scenario.breakdown_penalty,
                "exit_payoff_mode": exit_payoff_mode,
                "method_label": method.method_label,
                "display_name": method.display_name,
                "method": method.method,
                "method_family": method.method_family,
                "ordering_variant": method.ordering_variant,
                "use_move_ordering": int(method.use_move_ordering),
                "include_in_main_plots": int(method.include_in_main_plots),
                "depth_limit": method.depth_limit if method.depth_limit is not None else "full",
                "heuristic": method.heuristic or "none",
                "baseline_method_label": minimax_method.method_label,
                "baseline_root_action_label": baseline_metrics["root_action_label"],
                "baseline_root_value": baseline_metrics["root_value"],
                "baseline_final_status": baseline_metrics["final_status"],
                "baseline_utility": baseline_metrics["utility"],
                "baseline_final_price": baseline_metrics["final_price"],
            }
            row.update(metrics)
            row["match_minimax_root_action"] = 1.0 if actions_equal(method_root_action, baseline_root_action) else 0.0
            row["match_minimax_root_action_type"] = (
                1.0 if action_type(method_root_action) == action_type(baseline_root_action) else 0.0
            )
            row["same_final_status_as_minimax"] = (
                1.0 if str(metrics["final_status"]) == str(baseline_metrics["final_status"]) else 0.0
            )
            row["utility_gap_vs_minimax"] = float(metrics["utility"]) - float(baseline_metrics["utility"])
            row["absolute_utility_gap_vs_minimax"] = abs(float(row["utility_gap_vs_minimax"]))
            row["price_gap_vs_minimax"] = price_gap
            row["absolute_price_gap_vs_minimax"] = abs(price_gap) if not math.isnan(price_gap) else float("nan")
            row["root_value_gap_vs_minimax"] = method_root_value - baseline_root_value
            row["absolute_root_value_gap_vs_minimax"] = abs(float(row["root_value_gap_vs_minimax"]))
            row["node_ratio_vs_minimax"] = method_nodes / baseline_nodes if baseline_nodes else float("nan")
            row["node_reduction_vs_minimax"] = 1.0 - float(row["node_ratio_vs_minimax"]) if baseline_nodes else float("nan")
            row["runtime_ratio_vs_minimax"] = method_runtime / baseline_runtime if baseline_runtime else float("nan")
            row["runtime_reduction_vs_minimax"] = (
                1.0 - float(row["runtime_ratio_vs_minimax"]) if baseline_runtime else float("nan")
            )

            rounded_row: Dict[str, object] = {}
            for key, value in row.items():
                if isinstance(value, float):
                    rounded_row[key] = round(value, 6)
                else:
                    rounded_row[key] = value
            raw_rows.append(rounded_row)

    summary_by_method = summarize_rows(raw_rows, group_keys=["method_label", "display_name"])
    summary_by_class_method = summarize_rows(
        raw_rows,
        group_keys=["scenario_class", "method_label", "display_name"],
    )
    summary_move_ordering = summarize_rows(
        [row for row in raw_rows if row["ordering_variant"] in {"ordered", "unordered"}],
        group_keys=["method_family", "ordering_variant"],
    )
    summary_vs_minimax = summarize_rows(raw_rows, group_keys=["method_label", "display_name"])
    return raw_rows, summary_by_method, summary_by_class_method, summary_move_ordering, summary_vs_minimax



def print_console_summary(summary_rows: Sequence[Dict[str, object]], title: str, max_rows: int = 12) -> None:
    print(f"\n=== {title} ===")
    for idx, row in enumerate(summary_rows):
        if idx >= max_rows:
            print(f"... ({len(summary_rows) - max_rows} more rows)")
            break
        parts = [f"{key}={value}" for key, value in row.items()]
        print(" | ".join(parts))



def main() -> None:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(base_dir, exist_ok=True)

    (
        raw_rows,
        summary_by_method,
        summary_by_class_method,
        summary_move_ordering,
        summary_vs_minimax,
    ) = run_batch_experiments()

    raw_path = os.path.join(base_dir, "negotiation_batch_raw_results.csv")
    method_path = os.path.join(base_dir, "negotiation_batch_summary_by_method.csv")
    class_method_path = os.path.join(base_dir, "negotiation_batch_summary_by_class_method.csv")
    move_ordering_path = os.path.join(base_dir, "negotiation_batch_summary_move_ordering.csv")
    vs_minimax_path = os.path.join(base_dir, "negotiation_batch_summary_vs_minimax.csv")

    write_csv(raw_path, raw_rows)
    write_csv(method_path, summary_by_method)
    write_csv(class_method_path, summary_by_class_method)
    write_csv(move_ordering_path, summary_move_ordering)
    write_csv(vs_minimax_path, summary_vs_minimax)

    print(f"Wrote raw results to: {raw_path}")
    print(f"Wrote summary by method to: {method_path}")
    print(f"Wrote summary by class & method to: {class_method_path}")
    print(f"Wrote move-ordering summary to: {move_ordering_path}")
    print(f"Wrote vs-minimax summary to: {vs_minimax_path}")

    print_console_summary(summary_by_method, "Summary by method")
    print_console_summary(summary_move_ordering, "Move-ordering summary")


if __name__ == "__main__":
    main()
