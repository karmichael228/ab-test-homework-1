"""Однократная генерация данных по условию задания; существующий CSV не заменяется."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd


def generate_once(directory=Path(__file__).resolve().parent):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "experiment_data.csv"
    if csv_path.exists():
        raise FileExistsError("CSV уже существует: повторная генерация запрещена.")

    # Один случайный seed, затем строго исходная последовательность вызовов RNG.
    my_seed = int(np.random.randint(5000000, size=1)[0])
    np.random.seed(my_seed)
    n_users = int(np.random.randint(1000, 10000, size=1)[0])
    pre_activity = np.random.exponential(scale=10, size=n_users).astype(int)
    variant = np.random.choice(["control", "test"], size=n_users, p=[0.4, 0.6])
    prob_test = 0.4 + 0.3 * (pre_activity / pre_activity.max())
    variant2 = np.where(np.random.random(n_users) < prob_test, "test", "control")
    # Сохраняем перезапись из задания, включая предыдущий расход случайных чисел.
    variant = variant2
    error_flag = np.zeros(n_users)
    error_flag[variant == "control"] = np.random.binomial(
        1, 0.01, sum(variant == "control")
    )
    error_flag[variant == "test"] = np.random.binomial(
        1, 0.02, sum(variant == "test")
    )
    conversion_prob = 0.05 + 0.01 * (pre_activity / pre_activity.max())
    converted = np.random.binomial(1, conversion_prob, n_users)
    start_date = pd.Timestamp("2026-05-01")
    dates = [start_date + pd.Timedelta(days=np.random.randint(0, 14))
             for _ in range(n_users)]
    # Единственное исправление исходника — синтаксическая опечатка `}]` -> `})`.
    df = pd.DataFrame({
        "user_id": range(n_users), "variant": variant, "converted": converted,
        "error_flag": error_flag, "pre_activity": pre_activity, "date": dates,
    })
    df.to_csv(csv_path, index=False)
    metadata = {
        "MY_SEED": my_seed, "n_users": n_users,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "generation_runs": 1,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__,
        "note": "Исходные параметры сохранены. Seed не подбирался по результатам тестов.",
    }
    (directory / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return df, metadata


if __name__ == "__main__":
    generate_once()
