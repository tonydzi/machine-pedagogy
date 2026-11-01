---
title: T5 pilot-dev v1, instrument validation
date: 2026-10-23
type: reference
stage: raw
status: PILOT_ONLY_NOT_CONFIRMATION
tags: [machine-pedagogy, t5, pilot]
---

# Первый пилот: техническая парность есть, интерпретация ограничена

Источник: девять неизменяемых JSON в `runs/pilot-dev/`, seeds0,1,2. Статистика:
[pilot-summary.json](pilot-summary.json). Исходный source3da882c8…05a7a70 и manifest
baf2500e…77142 сохранены в Git `333a7e8459` вместе с результатами.
[README](README.md), [PREREGISTRATION](PREREGISTRATION.md), [DEV-AMENDMENT-2026-10-23](DEV-AMENDMENT-2026-10-23.md).

## Наблюдения

| Среднее трёх pilot seeds | fixed | permuted | variable |
|---|---:|---:|---:|
| A native до B | 1.0000 | 1.0000 | 1.0000 |
| A common до B | .1302 | .1628 | .4023 |
| A common после B | .1445 | .1289 | .3633 |
| B common, H1 | .1289 | .1367 | .3607 |
| C raw learning AUC | .6089 | .6107 | .7515 |
| C composition после | .7969 | .6432 | .8672 |

Native B=1.0 в восьми runs; variable seed2=.69921875. Это наблюдение,
не основание исключить seed или подбирать новое обучение.
Во всех трёх парных seeds совпали реальные semantic minibatch hashes,
data/config manifest и реальные tensor inputs фазы C между всеми плечами.

Общий A до B у постоянных форм близок к majority baseline, хотя native A=1.
Следовательно малая common delta не доказывает хорошую память освоенного A.
Common H2 остаётся измеримым изменением точности на другой сериализации,
но этот пилот не отвечает на вопрос о потере уже освоенного common A.

H1 различает плечи, однако variable встречало перестановки полей в обучении.
Результат относится к переносу вместе с аугментацией и новизной порядка у fixed,
не выделяет универсальную эргономику формы. Ни направление эффекта, ни новизна
не установлены подтверждающими данными.

## Цена и дисперсия

Каждый run:610560tokens,40704examples,593forward,550backward. Категории ledger
совпадают по плечам; отдельный hook проверил реальные forward calls/tokens.
Сумма внутренних wall times270.9435403001262s, среднее30.1048378111s,
диапазон19.8363–73.7042s. Driver child elapsed294.7970000002533s учитывает
запуск процессов и составляет использованную часть общего cap1800s.
CPU process sum191.94s; энергия и FLOPs не измерялись.

Fixed seed2 имеет73.7s wall, из них55.51s preprocessing. Причина выброса UNKNOWN;
OS wait/нагрузка возможны, не доказаны. Порядок запуска arms постоянный, поэтому
причинное сравнение скорости форм по wall time здесь недопустимо.

H1 парные контрасты variable−mean(fixed,permuted):
0.21875,0.26953125,0.1953125; среднее0.2278645833, SD0.0379395893.
Это три разведочные единицы после просмотра результатов, не confirmation.
Пилотный t-CI и p сохранены в JSON, не являются независимым подтверждением.

Для проектных3п.п. normal approximation предлагает n13, но exact noncentral-t
при той же SD даёт около74.46% мощности для n13. Для n18 условно около88.5%,
но SD из трёх seeds нестабильна; остальные mapping ещё не наблюдались.
План18 fresh seeds20–37 обеспечивает по два повтора девяти mapping/position cells;
его честная маркировка pilot-scale/finite precision, без гарантии80% power.
Ни один fresh seed пока не обучался. Окончательный freeze и exact-target
Codex+Grok ACCEPT обязательны до любого такого запуска.

## Исправление измерения, до новых данных

[DEV-AMENDMENT-2026-10-23](DEV-AMENDMENT-2026-10-23.md) добавляет только operational native accuracy change
в отдельной dev-ревизии, с прежним обучением и сохранённым исходным H2.
Общий лимит пилота сохраняется. Ревизия v2 завершена9/9; все прежние метрики,
curve accuracy/loss, data/config, actual semantic/C-input hashes совпали с v1
побитно. Новые actual input/label hashes парны pre/post, weight hashes before=after
и Torch RNG unchanged во всех9. Живой budget каждого run614400tokens/40960examples/
595forward/550backward; единственное изменение счётчиков3840tokens/256examples/
2evalforward/0backward.

V2 internal wall sum135.7200098001631s, CPU process sum132.546875s,
driver elapsed151.12199999997392s. Вместе v1+v2 driver445.919s<1800.
Быстрее v1 по wall не означает эффект формы: это повтор на другом состоянии хоста.
Source+outputs v2 сохранены backupac696cc13b. Все исходные v1 bytes сохранены.

Native A post-B по seeds0/1/2:

| Плечо | seed0 | seed1 | seed2 |
|---|---:|---:|---:|
| fixed | .15234375 | .16015625 | .20703125 |
| permuted | .15234375 | .16015625 | .20703125 |
| variable | .1640625 | .16015625 | .30078125 |

Pre=1.0 во всех9. Таблица показывает observed operational change отдельно
по каждому плечу; native cross-arm contrast/ranking не вычисляется. Variableseed2
имеет B-native=.69921875, ниже per-run gate. Его данные сохранены с флагом,
без чтения как loss-after-acquired-B. Ни одна строка не исключена.
Common H2 остаётся floor-limited; новая колонка не заменяет его estimand.

Проверка вторым методом: `audit_v2_results.py` exit0, receipt
`<local>/v2-parity-receipt.json`.
V1 не записывал actual eval-tensor hashes; не приписываем ему эти данные задним
числом. V2 записывает и проверяет парность собственных actual eval bytes.
