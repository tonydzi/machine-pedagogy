---
title: T5 form-cost, results of the frozen 18-seed series
date: 2026-10-23
type: reference
stage: raw
status: artifact.proposed
tags: [machine-pedagogy, t5, results]
---

# T5: результат при одинаковом бюджете обучения

В этой синтетической задаче variable дал лучший перенос B на общую четвёртую
форму: +22.58 процентного пункта относительно среднего fixed/permuted.
При этом оба постоянных плеча ниже простого majority baseline 16.2543%;
только variable выше него и заранее заданного floor threshold 26.2543%.
Контраст отражает перенос переменной формы при провале переноса постоянных.
Все 18 свежих seed contrasts положительны. Это результат конкретной стратегии
обучения и переноса формы; отдельный механизм эргономики, latent forgetting,
model welfare и научная новизна этим опытом не установлены.

Полный набор: seeds 20–37, 18 парных единиц,54/54 complete. Исключений 0,
все 54 запущены по одному разу. Оба dev набора seeds 0–2 сохранены и в inference
не входят. Правила и исходы заданы до fresh данных в [PREREGISTRATION-v2](PREREGISTRATION-v2.md),
включающем immutable [PREREGISTRATION](PREREGISTRATION.md). История: [README](README.md), [pilot-analysis](pilot-analysis.md),
[ANALYSIS-ADDENDUM-2026-10-23](ANALYSIS-ADDENDUM-2026-10-23.md). Итоговый экран: final-status-v2.html (not exported).

## Первичный H1

| Показатель | Результат |
|---|---:|
| Fixed, common B accuracy | 12.4349% |
| Permuted, common B accuracy | 13.8021% |
| Variable, common B accuracy | 35.6988% |
| Mean B-test majority baseline | 16.2543% |
| Preregistered global-floor threshold | 26.2543% |
| Парный contrast variable − mean(fixed,permuted) | +22.5803 п.п. |
| 95% t-интервал, df 17 | [+18.9567; +26.2039] п.п. |
| Двухсторонний p | 2.45950522350604e-10 |
| SD 18 парных contrasts | 7.28675 п.п. |
| Sample variance contrasts | 0.005309672137491064 accuracy² |

Это единственный первичный тест, alpha=.05. N не менялся после просмотра
результатов. Все 18 contrasts, seed-level данные и исходные JSON сохранены в
`final-summary.json`; независимый расчёт stdlib/SciPy совпал по всем18 разностям,
mean, CI, p и feasibility. p в независимом расчёте 2.45950522350605e-10,
отличие в последнем округлении, допуск сверки 1e-12 relative/1e-14 absolute.

По заранее заданным общим feasibility gates `assay_limited=false`: native A
до B=1.0 во всех 54; ни одно arm-mean B-native не ниже .80; флаг общего H1 пола
не сработал, потому что он требует низких средних сразу у всех трёх плеч.
Fixed 12.4349% и permuted 13.8021% ниже majority 16.2543% и порога 26.2543%;
variable 35.6988% выше обоих. Это выход одного плеча выше простого ориентира
при двух неуспешных постоянных переносах. `H1_global_floor=false` не означает,
что каждое плечо вышло из пола. Потолка нет. Остальные ограничения сохраняются.

## Сохранение A и обучаемость

Исходный common H2 остаётся в отчёте: все 54 common A-pre ниже .80,
наблюдаемый диапазон.109375–.421875. Поэтому эта проба не измеряет сохранение
уже освоенного на ней навыка. Её изменение не выдаётся за отсутствие забывания.

Новая native A-проба сохраняет одинаковые input/label bytes до/после B,
не меняет веса/Torch RNG при измерении и читается отдельно для каждого run.
Межплечевых native contrasts, среднего изменения A-native, pooling и Pareto
ranking по native результатам нет. Без no-B control и других диагностик она
не устанавливает latent forgetting или изолированный причинный вред B.
Низкая native A-post точность также может соответствовать константному ответу
около label-majority пола этой пробы. A-native majority baseline здесь отдельно
не оценён. Поэтому даже eligible строка не доказывает измерение остаточной
скрытой информации; native result остаётся операционным per-run наблюдением.

Порог native-retention eligibility выполнен в 50/54 строках. Четыре variable
run не достигли B-native .80; они сохранены во всех данных и H1 inference:

| Seed | B-native | Статус |
|---|---:|---|
| 23 | .4921875 | interpretation flag, retained |
| 25 | .53125 | interpretation flag, retained |
| 28 | .40234375 | interpretation flag, retained |
| 37 | .72265625 | interpretation flag, retained |

## Последующее обучение C, описательно

Все показатели ниже заданы заранее; дополнительных p-values и выбора
успешных checkpoint/seed нет. C-AUC нормирован по шагам 0–150 на C-dev.

| Плечо | C-dev начало | C-dev конец | C-dev AUC | C-comp до | C-comp после | C-comp gain |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 12.2396% | 82.8993% | .629196 | 10.9375% | 67.0573% | +56.1198 п.п. |
| Permuted | 11.6319% | 87.5868% | .653646 | 11.4583% | 73.6111% | +62.1528 п.п. |
| Variable | 35.5035% | 92.2309% | .773474 | 27.8212% | 85.7205% | +57.8993 п.п. |

Средние C-comp baselines на тех же frozen records: majority 38.4115%,
oracle-copy A 50.0434%, oracle-copy B 49.9566%. Отдельные C-dev/comp baselines,
endpoints и результаты для всех 54 строк есть в `feasibility.per_run` итогового
JSON; на экране показаны их сводные значения. Начальные C уровни отличаются, поэтому финальная точность
и AUC не заменяют отдельную оценку прироста. Усреднение не доказывает успех
каждого run или общее композиционное рассуждение за пределами этой задачи.
Все три средних C-comp pre ниже majority baseline 38.4115%; часть gain начинается
с уровня ниже простого ориентира. Отдельные post результаты тоже могут оставаться
ниже своего majority baseline; их per-run строки доступны в JSON. Например,
seed 24/fixed: C-comp post 42.1875% при своём majority 44.53125%.

## Цена измеренного вычисления

| Набор | Runs | Tokens | Forward | Backward | Сумма CPU seconds | Сумма wall внутри core, s |
|---|---:|---:|---:|---:|---:|---:|
| Dev-v1 | 9 | 5,495,040 | 5,337 | 4,950 | 191.937500 | 270.943540 |
| Dev-v2 | 9 | 5,529,600 | 5,355 | 4,950 | 132.546875 | 135.720010 |
| Fresh | 54 | 33,177,600 | 32,130 | 29,700 | 899.890625 | 929.774115 |
| Все измеренные model runs | 72 | 44,202,240 | 42,822 | 39,600 | 1,224.375000 | 1,336.437665 |

Fresh на run: 614400 tokens,40960 examples,595 forwards,550 backwards.
Добавка native-пробы 3840 tokens/256 examples/2 forwards/0 backwards.
Бюджет между плечами одинаков. Это число входных позиций в model forwards,
включая оценку, не счёт API-токенов коммерческой LLM. Повторные запуски
аналитического калькулятора и LLM-рецензии в model ledger не входят.
Padding/restoration=0; FLOPs, энергия, деньги и ресурсная цена хранения весов
не измерялись. Из результата не следует, что для заданной целевой точности
variable требует на22.58% меньше вычисления.

Fresh driver elapsed 1032.768s; он включает запуск/ожидание child process,
но исключает CPU waiting. Общий dev driver elapsed 445.919s <1800s.
Начало fresh driver 14:34:03 UTC, завершение 15:18:53 UTC, 44m50s по часам.
Два CPU defer были ожиданием, не нулевыми результатами. Во всех 54 успешных
стартах три CPU samples прошли порог 70%; threads 1/BelowNormal/CPU-only.
T3, GPU-настройки и чужие задачи этой работой не менялись.

| Плечо,18 runs | Mean core wall, s | SD wall, s | Range wall, s | Sum CPU, s |
|---|---:|---:|---|---:|
| Fixed | 16.666747 | 1.520878 | 13.958586–19.851615 | 291.484375 |
| Permuted | 17.641085 | 2.546625 | 14.501873–25.623983 | 305.765625 |
| Variable | 17.346285 | 2.494973 | 13.658282–22.393125 | 302.640625 |

Это наблюдаемое время на загруженной Windows-машине, не доказанный speed effect.
Pilot-v1 fixed/seed2 имел необычно длинную preprocessing wall; причина UNKNOWN,
она не объявляется свойством формы записи.

## Проверено и не проверено

Перед первым fresh run получены новые exact-target Codex+Grok ACCEPT через
`llm_ask --class second_opinion --mode all`; оба target hash повторили.
Первая панель Codex REQUEST_CHANGES/Grok ACCEPT сохранена; её согласие не
переиспользовалось после исправления H3 report. Claude integration APPROVE
и17 tests PASS относятся к текущему analyzer e18de596…; source 396313… неизменен.

Повторное чтение всех 9 frozen targets15:19:50 UTC, непосредственно перед inference,
совпало с gate/panel. Actual semantic minibatch hashes, C training inputs,
common/native A probe hashes,25 dev curve inputs, weight/RNG audit, frozen data
и budgets проверены для полного54 набора; оба admission метода rc0.
H1 tensor hash в live result не сохранён. Его одинаковость между плечами
реконструирована из frozen source/data/RNG, это expected-byte evidence,
а не отсутствующая live telemetry. Все saved raw JSON остаются неизменными.

Масштаб остаётся пилотным: 18 seeds дают конечное условное сравнение одной
архитектуры/задачи. Нет гарантии 80% мощности из трёх dev seeds. H1 смешивает
перенос формы, аугментацию и знакомство с перестановками; эргономику отдельно
не выделяет. Две постоянные формы не видели такой перестановочной вариативности.
Ни внешняя регистрация, ни публикация не проводились.

## Воспроизводимое чтение результата

Все пути ниже абсолютные относительно этого Windows узла:
`t5-form-cost/`.

Фактические команды:

```text
python -B <T5>/runs/driver.py --manifest <T5>/runs/confirmatory-manifest.json --check
rc0: valid=true, complete=true
python -B <T5>/runs/analyze.py --manifest <T5>/runs/confirmatory-manifest.json --out <T5>/final-summary.json --check
rc0: valid=true, runs54, paired_seeds18, inference=false
python -B <local>/operator_frozen_check.py
rc0: все9 SHA совпали, reviewers Codex+Grok, runs54
python -B <T5>/runs/analyze.py --manifest <T5>/runs/confirmatory-manifest.json --out <T5>/final-summary.json
rc0: valid=true, published=true, paired_seeds18
python -B <local>/capture_independent.py
rc0: valid=true, runs54, inference=true, primary_and_independent_agree=true
```

`<T5>` в компактной записи означает абсолютный корень выше, не исполняемый флаг.
`capture_independent.py` является обёрткой: внутри он запускает фактический
`verify_completed54_independent.py --manifest <T5>/runs/confirmatory-manifest.json`.
SHA256 этого независимого вычислителя
`77167febcf34cc4c0c44611d19f6b4d11d43a0b951af1fad7d9157efe649511f`;
именно этот внутренний command сохранён в independent receipt.
Read-only `--check` не пересчитывает inference. Итоговый publisher переиспользует
только побитно идентичный JSON и отказывает перезаписи иного содержимого.

| Артефакт | SHA256 |
|---|---|
| `final-summary.json` | `836b43ef3f0d45a21b3ddece68e0d27f004ec1a9b545255c9f43060e3bd45a79` |
| `runs/confirmation-independent.json` | `fc05e1627876ddf6e12c7cbcd741564a248458eee65e042806eb4447f177aa6c` |
| `runs/confirmatory-gate.json` | `8c9a68ecaf9feedc21532b7bf02e89d4b20e7a181c8a475b1d5f0a90191fbe1b` |

Полные source/config/data/base/active/analysis/driver/watch hashes находятся
в gate.frozen_targets и обоих panel evidence. Проверка итогового доклада:
final-result-acceptance (not exported).

Первое ревью доклада: Claude REQUEST_CHANGES, arithmetic confirmed,
`review-20261023-162824.md`. Уточнения выше добавлены до окончательного вердикта:
baseline провала постоянных H1, возможный native floor и C-pre baseline.
Ни raw JSON, ни frozen analysis, ни статистические результаты не менялись.
