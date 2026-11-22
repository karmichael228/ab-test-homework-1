"""Аудит A/B-теста. Запуск: python analyze_experiment.py --data experiment_data.csv.

Анализ не перегенерирует данные. Статистические решения используют alpha=0.05.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ALPHA = 0.05
GROUPS = ["control", "test"]
COLORS = ["#376DA0", "#E18A38"]


def load_data(path):
    """Останавливаем анализ при нарушении структуры, без молчаливой очистки."""
    df = pd.read_csv(path)
    required = {"user_id", "variant", "converted", "error_flag", "pre_activity", "date"}
    if not required.issubset(df.columns):
        raise ValueError(f"Нет колонок: {required - set(df.columns)}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("Пустая таблица или пропуски в обязательных колонках")
    if df.user_id.duplicated().any():
        raise ValueError("Повторные user_id: нарушена единица рандомизации")
    if set(df.variant) != set(GROUPS):
        raise ValueError("Ожидаются две группы: control, test")
    for column in ["converted", "error_flag"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} должна быть бинарной")
    if not pd.api.types.is_numeric_dtype(df.pre_activity):
        raise ValueError("pre_activity должна быть числовой")
    if not np.isfinite(df.pre_activity).all() or (df.pre_activity < 0).any():
        raise ValueError("pre_activity содержит недопустимые значения")
    df["date"] = pd.to_datetime(df.date, errors="raise")
    if min(df.groupby("variant").size()) < 2:
        raise ValueError("В каждой группе нужно не менее двух пользователей")
    return df


def summary_table(df):
    result = df.groupby("variant").agg(
        n=("user_id", "size"), conversions=("converted", "sum"),
        conversion_rate=("converted", "mean"), errors=("error_flag", "sum"),
        error_rate=("error_flag", "mean"), activity_mean=("pre_activity", "mean"),
        activity_median=("pre_activity", "median"), activity_std=("pre_activity", "std"),
    ).reindex(GROUPS)
    result["share"] = result.n / len(df)
    return result


def wilson_interval(successes, total, alpha=ALPHA):
    """95% Wilson CI для доли: устойчивее Wald CI при редких ошибках."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denominator
    half = z / denominator * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return float(center - half), float(center + half)


def rate_difference(df, metric, compute_test=False):
    """Разность test-control и независимый Newcombe CI на основе Wilson CI."""
    s = df.groupby("variant")[metric].agg(["sum", "count", "mean"]).reindex(GROUPS)
    pc, pt = s.loc["control", "mean"], s.loc["test", "mean"]
    lc, uc = wilson_interval(s.loc["control", "sum"], s.loc["control", "count"])
    lt, ut = wilson_interval(s.loc["test", "sum"], s.loc["test", "count"])
    delta = float(pt - pc)
    lower = delta - np.sqrt((pt - lt) ** 2 + (uc - pc) ** 2)
    upper = delta + np.sqrt((ut - pt) ** 2 + (pc - lc) ** 2)
    result = {"control_rate": float(pc), "test_rate": float(pt), "difference": delta,
              "ci_low": float(lower), "ci_high": float(upper)}
    if compute_test:
        table = [[int(s.loc[g, "sum"]), int(s.loc[g, "count"] - s.loc[g, "sum"])]
                 for g in GROUPS]
        result["p_value"] = float(stats.fisher_exact(table, alternative="two-sided").pvalue)
    return result


def holm_adjust(p_values):
    """Поправка Holm для трёх заранее выбранных диагностик (дополнительно)."""
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def audit(df, design_eligible=True):
    """design_eligible=False для искусственно отобранной наблюдательной выборки."""
    summary = summary_table(df)
    counts = summary.n.to_numpy()
    srm = stats.chisquare(counts, f_exp=np.repeat(len(df) / 2, 2))
    c = df.loc[df.variant == "control", "pre_activity"]
    t = df.loc[df.variant == "test", "pre_activity"]
    # Основная проверка распределений: асимптотический U-тест с поправкой на ties.
    u = stats.mannwhitneyu(t, c, alternative="two-sided", method="asymptotic")
    welch = stats.ttest_ind(t, c, equal_var=False)
    scale = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    smd = float((t.mean() - c.mean()) / scale) if scale > 0 else 0.0
    guard = rate_difference(df, "error_flag", compute_test=True)
    tests = pd.DataFrame({
        "check": ["SRM: chi-square, 50/50", "pre_activity: Mann–Whitney U", "error_flag: Fisher"],
        "p_value": [float(srm.pvalue), float(u.pvalue), guard["p_value"]],
    }).set_index("check")
    tests["p_holm"] = holm_adjust(tests.p_value)
    tests["warning_alpha_0.05"] = tests.p_value < ALPHA
    warnings = []
    if srm.pvalue < ALPHA:
        warnings.append("SRM обнаружен")
    if u.pvalue < ALPHA:
        warnings.append("Дисбаланс по pre_activity — возможен confounder")
    if guard["p_value"] < ALPHA:
        direction = "ухудшение" if guard["difference"] > 0 else "изменение"
        warnings.append(f"Дисбаланс guardrail error_flag — {direction}")
    result = {
        "n": len(df), "srm_chi2": float(srm.statistic), "srm_p": float(srm.pvalue),
        "activity_u": float(u.statistic), "activity_p": float(u.pvalue),
        "activity_welch_p": float(welch.pvalue), "activity_smd": smd,
        "guardrail": guard, "warnings": warnings, "checks_passed": not warnings,
        "valid": not warnings and design_eligible,
    }
    # По условию стандартный анализ конверсии разрешён только после всех проверок.
    if result["valid"]:
        result["primary_analysis"] = rate_difference(df, "converted", compute_test=True)
    return summary, tests, result


def exact_match(df, seed=20261122):
    """1:1 отбор в каждом точном значении pre_activity, без использования исходов.

    Сохранение исходных user_id позволяет проверить отсутствие повторов/подмены.
    Это балансировка наблюдаемой ковариаты, а не восстановление рандомизации.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for _, stratum in df.groupby("pre_activity", sort=True):
        ids = {g: stratum.index[stratum.variant == g].to_numpy() for g in GROUPS}
        k = min(len(ids[g]) for g in GROUPS)
        if k:
            for group in GROUPS:
                selected.extend(rng.choice(ids[group], size=k, replace=False))
    if not selected:
        raise ValueError("Нет общего покрытия по pre_activity")
    return df.loc[selected].copy().sort_values("user_id").reset_index(drop=True)


def plot_diagnostics(df):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    summary = summary_table(df)
    ax = axes[0, 0]
    bars = ax.bar(GROUPS, summary.n, color=COLORS, width=0.55)
    ax.axhline(len(df) / 2, linestyle="--", color="#666666", label="Ожидание 50/50")
    ax.bar_label(bars, labels=[f"{n:,}\n({n / len(df):.1%})" for n in summary.n], padding=5)
    ax.set(title="1. Размеры групп", ylabel="Пользователи", ylim=(0, summary.n.max() * 1.25))
    ax.legend(loc="upper right")
    ax = axes[0, 1]
    for group, color in zip(GROUPS, COLORS):
        x = np.sort(df.loc[df.variant == group, "pre_activity"].to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=group, color=color)
    ax.set(title="2. Распределение активности (ECDF)", xlabel="pre_activity", ylabel="Накопленная доля")
    ax.legend()
    ax = axes[1, 0]
    for position, (group, color) in enumerate(zip(GROUPS, COLORS)):
        row = summary.loc[group]
        lo, hi = wilson_interval(row.errors, row.n)
        ax.errorbar(position, 100 * row.error_rate,
                    yerr=[[100 * (row.error_rate - lo)], [100 * (hi - row.error_rate)]],
                    fmt="o", capsize=6, markersize=8, color=color)
        ax.annotate(f"{int(row.errors)}/{int(row.n)}", (position, 100 * hi),
                    xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set(xticks=[0, 1], xticklabels=GROUPS, xlim=(-0.6, 1.6),
           title="3. Ошибки страницы: доли и 95% Wilson CI", ylabel="Частота ошибок, %")
    ax.margins(y=0.25)
    ax = axes[1, 1]
    bins = pd.qcut(df.pre_activity, q=5, duplicates="drop")
    trend = df.assign(activity_bin=bins, in_test=(df.variant == "test").astype(int)).groupby(
        "activity_bin", observed=True).agg(n=("in_test", "size"), test=("in_test", "sum"))
    p = trend.test / trend.n
    ci = np.array([wilson_interval(k, n) for k, n in zip(trend.test, trend.n)])
    ax.errorbar(np.arange(len(trend)), p, yerr=[p - ci[:, 0], ci[:, 1] - p],
                fmt="o-", color=COLORS[1], capsize=5)
    ax.axhline(0.5, linestyle="--", color="#666666", label="Независимое назначение 50/50")
    labels = [f"{interval.left:g}…{interval.right:g}\nn={n}" for interval, n in zip(trend.index, trend.n)]
    ax.set(xticks=np.arange(len(trend)), xticklabels=labels, ylim=(0.2, 0.75),
           title="4. Доля test по квантильным интервалам активности", ylabel="Доля test (95% Wilson CI)")
    ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("Проверка дизайна эксперимента", fontsize=18, fontweight="bold")
    return fig


def plot_correction(df, matched):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for data, title, ax in zip([df, matched], ["Исходные данные", "Точное сопоставление по активности"], axes):
        summary = summary_table(data)
        for pos, (group, color) in enumerate(zip(GROUPS, COLORS)):
            row = summary.loc[group]
            lo, hi = wilson_interval(row.conversions, row.n)
            ax.errorbar(pos, row.conversion_rate * 100,
                        yerr=[[100 * (row.conversion_rate - lo)], [100 * (hi - row.conversion_rate)]],
                        fmt="o", markersize=8, capsize=6, color=color)
        ax.set(xticks=[0, 1], xticklabels=GROUPS, xlim=(-0.6, 1.6),
               title=f"{title}\nn={len(data):,}", ylabel="Конверсия, %; 95% Wilson CI")
    limits = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
    fig.suptitle("Описательная чувствительность конверсии к балансировке", fontsize=15)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("experiment_data.csv"))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "analysis_outputs")
    args = parser.parse_args()
    df = load_data(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    summary, tests, result = audit(df)
    print("Базовая информация\n", summary.to_string())
    print("\nПроверки дизайна\n", tests.to_string())
    print(f"\nSMD активности = {result['activity_smd']:.4f}; дополнительный Welch p = {result['activity_welch_p']:.6g}")
    print("Guardrail (разность test-control):", result["guardrail"])
    for warning in result["warnings"]:
        print("ПРЕДУПРЕЖДЕНИЕ:", warning)
    if result["valid"]:
        p = result["primary_analysis"]["p_value"]
        print("Проверки пройдены. Конверсия:", "значимые различия" if p < ALPHA else "значимых различий не обнаружено")
    else:
        print("Эксперимент не валиден. Остановить тест, исправить ошибки, перезапустить")
        print("Стандартный анализ конверсии заблокирован; ниже только описательная чувствительность.")
    matched = exact_match(df)
    _, matched_tests, matched_result = audit(matched, design_eligible=False)
    print("\nБонус: осталось", len(matched), "пользователей; доля сохранения", len(matched) / len(df))
    print(matched_tests.to_string())
    descriptive = {label: {k: v for k, v in rate_difference(data, "converted").items() if k != "p_value"}
                   for label, data in [("before", df), ("after", matched)]}
    print("Конверсия: описательные оценки и приближённые CI", descriptive)
    print("Балансировка не восстанавливает рандомизацию и не устраняет причины SRM/ошибок; нужен перезапуск.")
    summary.to_csv(args.out / "group_summary.csv")
    tests.to_csv(args.out / "design_checks.csv")
    matched.to_csv(args.out / "matched_data.csv", index=False)
    result["bonus"] = {"matched_n": len(matched), "audit": matched_result, "conversion_descriptive": descriptive}
    (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, fig in [("diagnostics", plot_diagnostics(df)), ("correction", plot_correction(df, matched))]:
        fig.savefig(args.out / f"{name}.png", dpi=150, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
