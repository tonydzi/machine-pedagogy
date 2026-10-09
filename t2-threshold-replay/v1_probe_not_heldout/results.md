# T2 results

config: {"S": 300, "N": 5, "bs": 32, "lr": 0.1, "hidden": 128, "m_replay": 20, "m_probe": 10}; tau grid [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]; tuned tau {'split': 0.2, 'perm': 0.2}; final seeds 0-4 on test; runtime 549s; torch 2.11.0+cu128

| bench | arm | final acc (mean ± sd) | BWT (mean ± sd) | replay steps | diag forwards |
|---|---|---|---|---|---|
| perm | A | 0.508 ± 0.021 | -0.543 ± 0.013 | 0 ± 0 | 0 |
| perm | B | 0.766 ± 0.013 | -0.213 ± 0.030 | 240 ± 0 | 0 |
| perm | C | 0.759 ± 0.022 | -0.219 ± 0.027 | 240 ± 0 | 325 |
| perm | B_match | 0.764 ± 0.018 | -0.215 ± 0.036 | 240 ± 0 | 0 |
| perm | D_rand | 0.767 ± 0.018 | -0.210 ± 0.025 | 240 ± 0 | 0 |
| split | A | 0.197 ± 0.003 | -0.999 ± 0.002 | 0 ± 0 | 0 |
| split | B | 0.883 ± 0.021 | -0.132 ± 0.021 | 240 ± 0 | 0 |
| split | C | 0.876 ± 0.009 | -0.139 ± 0.009 | 240 ± 0 | 282 |
| split | B_match | 0.866 ± 0.017 | -0.156 ± 0.017 | 240 ± 0 | 0 |
| split | D_rand | 0.835 ± 0.057 | -0.193 ± 0.067 | 240 ± 0 | 0 |

red probe all pass: True
