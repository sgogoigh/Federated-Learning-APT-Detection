# LANL Federated APT — Experiment Summary

Seeds: [42, 7, 123] | non-IID host partition | 7 clients, 15 rounds

## Multi-seed (mean ± std)

| model | ROC-AUC | PR-AUC | TPR@0.1%FPR | TPR@0.01%FPR |
|---|---|---|---|---|
| no-graph MLP | 0.9011 ± 0.0086 | 0.0070 ± 0.0044 | 0.2212 ± 0.1501 | 0.0123 ± 0.0142 |
| centralized GNN | 0.9918 ± 0.0011 | 0.0740 ± 0.0142 | 0.7051 ± 0.0697 | 0.3088 ± 0.0489 |
| fed (edges) | 0.9051 ± 0.0117 | 0.0014 ± 0.0009 | 0.0860 ± 0.0600 | 0.0123 ± 0.0142 |
| fed (positives) | 0.9853 ± 0.0027 | 0.0808 ± 0.0357 | 0.7235 ± 0.0434 | 0.2273 ± 0.1680 |
| fed (val_signal) | 0.9775 ± 0.0095 | 0.0891 ± 0.0376 | 0.7573 ± 0.0586 | 0.2980 ± 0.1092 |

## pos_smooth sweep (seed 42, positive-aware)

| pos_smooth | ROC-AUC | PR-AUC | TPR@0.1%FPR |
|---|---|---|---|
| 1 | 0.9820 | 0.0452 | 0.7051 |
| 10 | 0.9749 | 0.0408 | 0.5115 |
| 100 | 0.9383 | 0.0044 | 0.1982 |
| 1000 | 0.9070 | 0.0011 | 0.0968 |
