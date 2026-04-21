# Deep Momentum

Исследование ML и DL подходов к cross-sectional momentum-стратегиям на top-3000 американских акций. Проект реализует унифицированный walk-forward пайплайн для обучения моделей, построения long/short-портфеля с хеджированием рыночной экспозиции и замера стандартных портфельных метрик (Sharpe, Alpha, IR, Max Drawdown, Turnover).

## Структура репозитория

```
deep-momentum/
├── backtests/
│   └── deep_momentum/
│       ├── deep_momentum/
│       │   ├── config/              # конфиги эксперимента и торговли
│       │   ├── strategies/          # реализации стратегий
│       │   │   ├── features.py
│       │   │   ├── systematic_momentum.py      # наивный momentum
│       │   │   ├── ml_momentum.py              # линейная регрессия
│       │   │   ├── logistic_momentum.py        # L1-logistic
│       │   │   ├── mlp_classifier_momentum.py  # MLP-классификатор
│       │   │   └── lstm_momentum.py            # LSTM с MACD-фичами
│       │   └── run.py
│       ├── backtests/               # Jupyter-ноутбуки с бэктестами
│       │   ├── 0_Unhedged_Momentum.ipynb
│       │   ├── 1_Momentum.ipynb
│       │   ├── 2_First_Backtest.ipynb
│       │   ├── 3_MLP_classifie.ipynb
│       │   ├── 4_Logistic_Momentum.ipynb
│       │   ├── 5_MLP_Extended_Features.ipynb
│       │   ├── 6_MLP_Retrain_From_Scratch.ipynb
│       │   ├── 7_Ridge_Momentum.ipynb
│       │   ├── 8_MLP_Bigger.ipynb
│       │   ├── 9_MLP_MACD.ipynb
│       │   ├── 10_LSTM_Momentum.ipynb
│       │   └── backtest_ml.ipynb    # общий раннер для ML-моделей
│       └── pyproject.toml
├── classic_ml/
│   └── ml_metrics.ipynb             # ML-метрики (MSE, R², Accuracy, ROC-AUC)
├── data/                            # данные (в git не коммитится)
│   └── datasets/
│       ├── top3000_data_df.parquet
│       └── top3000_presence_matrix.parquet
├── requiements.txt                  # pinned-зависимости
└── README.md
```

## Установка

Требуется Python `3.12.5`.

```bash
python -m venv deep_venv
source deep_venv/bin/activate
pip install -r requiements.txt
pip install -e backtests/deep_momentum
```

Либо через `uv`:

```bash
uv venv
uv pip install -r requiements.txt
uv pip install -e backtests/deep_momentum
```

## Данные

В `data/datasets/` должны лежать два parquet-файла:

- `top3000_data_df.parquet` — дневные цены top-3000 тикеров + макро-факторы.
- `top3000_presence_matrix.parquet` — матрица присутствия тикера в универсе на дату (борьба с survivorship bias).

Данные в репозиторий не коммитятся (см. `.gitignore`).

## Стратегии

| Файл | Модель | Фичи |
|---|---|---|
| `systematic_momentum.py` | naive momentum (без обучения) | r1, r3, r6, r12 |
| `ml_momentum.py` | Linear/Ridge regression | momentum + vol |
| `logistic_momentum.py` | Logistic (L1, balanced) | r3, r6, r12 |
| `mlp_classifier_momentum.py` | MLP-классификатор | r3, r6, r12 |
| `lstm_momentum.py` | LSTM (seq_len=12) | momentum + 16 MACD/Signal-фич |

Все стратегии наследуются от `SortingStrategy` из `quant-pml` и поддерживают walk-forward переобучение, квантильный long/short и хеджирование бетой через `MarketFuturesHedge`.

## Запуск бэктеста

Основной раннер — `backtests/deep_momentum/backtests/backtest_ml.ipynb`. Меняется только переменная `MODEL` (например `"logistic"`, `"mlp"`, `"lstm"`), пайплайн сам собирает стратегию, гоняет walk-forward и печатает метрики.

ML-метрики (без бэктеста) считаются в `classic_ml/ml_metrics.ipynb` на out-of-time сплите (train до 2022, test 2022+).

## Метрики

- **ML:** MSE vs нулевой бейзлайн, R² на excess return, Accuracy на знаке, ROC-AUC.
- **Портфельные:** Sharpe, Alpha/Beta к S&P, Information Ratio, Volatility, Max Drawdown, Turnover.

## Требования к данным и протокол

- Causal features: все признаки на дату `t` считаются по данным строго до `t` (с учётом `exclude_td=21` — skip lag в 1 месяц для предотвращения краткосрочного реверсала).
- Walk-forward: модель переобучается на каждом месячном ребалансе на расширяющемся/фиксированном окне `train_months`.
- Portfolio construction: long top-quantile, short bottom-quantile, dollar-neutral, equally-weighted внутри ноги, с последующим hedge рыночного бета через фьючерс на SPX.
