# T2 results

config: {"S": 300, "N": 5, "bs": 32, "lr": 0.1, "hidden": 128, "m_replay": 20, "m_probe": 10}; tau grid [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]; tuned tau {'split': 0.2, 'perm': 0.1}; final seeds 0-4 on test; runtime 549s; torch 2.11.0+cu128

| bench | arm | final acc (mean ± sd) | BWT (mean ± sd) | replay steps | diag forwards |
|---|---|---|---|---|---|
| perm | A | 0.503 ± 0.011 | -0.552 ± 0.014 | 0 ± 0 | 0 |
| perm | B | 0.776 ± 0.033 | -0.206 ± 0.037 | 240 ± 0 | 0 |
| perm | C | 0.774 ± 0.028 | -0.209 ± 0.035 | 240 ± 0 | 315 |
| perm | B_match | 0.779 ± 0.036 | -0.201 ± 0.042 | 240 ± 0 | 0 |
| perm | D_rand | 0.775 ± 0.028 | -0.203 ± 0.029 | 240 ± 0 | 0 |
| split | A | 0.197 ± 0.003 | -0.999 ± 0.002 | 0 ± 0 | 0 |
| split | B | 0.902 ± 0.025 | -0.109 ± 0.031 | 240 ± 0 | 0 |
| split | C | 0.898 ± 0.030 | -0.113 ± 0.033 | 240 ± 0 | 326 |
| split | B_match | 0.884 ± 0.031 | -0.133 ± 0.036 | 240 ± 0 | 0 |
| split | D_rand | 0.863 ± 0.043 | -0.156 ± 0.054 | 240 ± 0 | 0 |

red probe all pass: True
