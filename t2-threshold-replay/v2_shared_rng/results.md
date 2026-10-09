# T2 results

config: {"S": 300, "N": 5, "bs": 32, "lr": 0.1, "hidden": 128, "m_replay": 20, "m_probe": 10}; tau grid [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]; tuned tau {'split': 0.2, 'perm': 0.2}; final seeds 0-4 on test; runtime 493s; torch 2.11.0+cu128

| bench | arm | final acc (mean ± sd) | BWT (mean ± sd) | replay steps | diag forwards |
|---|---|---|---|---|---|
| perm | A | 0.503 ± 0.011 | -0.552 ± 0.014 | 0 ± 0 | 0 |
| perm | B | 0.770 ± 0.014 | -0.213 ± 0.011 | 240 ± 0 | 0 |
| perm | C | 0.766 ± 0.025 | -0.215 ± 0.022 | 231 ± 21 | 354 |
| perm | B_match | 0.761 ± 0.015 | -0.224 ± 0.011 | 231 ± 21 | 0 |
| perm | D_rand | 0.748 ± 0.025 | -0.238 ± 0.022 | 231 ± 21 | 0 |
| split | A | 0.197 ± 0.003 | -0.999 ± 0.002 | 0 ± 0 | 0 |
| split | B | 0.892 ± 0.021 | -0.121 ± 0.027 | 240 ± 0 | 0 |
| split | C | 0.885 ± 0.017 | -0.131 ± 0.021 | 240 ± 0 | 283 |
| split | B_match | 0.871 ± 0.027 | -0.150 ± 0.036 | 240 ± 0 | 0 |
| split | D_rand | 0.846 ± 0.041 | -0.181 ± 0.050 | 240 ± 0 | 0 |

red probe all pass: True
