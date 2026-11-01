---
title: T5, active preregistration revision 2, analysis report completion
date: 2026-10-23
type: reference
stage: raw
status: FROZEN_BEFORE_FRESH_PENDING_EXACT_REVIEWS
frozen_utc: 2026-10-23T14:24:33Z
tags: [machine-pedagogy, t5, preregistration]
---

# T5: ревизия 2, завершение обещанного отчёта H3

Это тонкая ревизия, не копия всего протокола. Она включает по ссылке неизменный
[PREREGISTRATION](PREREGISTRATION.md) SHA256
`d51a14d0c38348ce3c2e5234c2e50665ee3146b4c71216c92882336069f7ee48`.
Все определения/правила base действуют, кроме явно заменённого ниже analysis SHA
и конкретизации исполнения уже заявленного H3. Обоснование и границы:
[ANALYSIS-ADDENDUM-2026-10-23](ANALYSIS-ADDENDUM-2026-10-23.md). Статус: [README](README.md). Fresh обучения ещё нет.

## Неизменные frozen targets

- Training source `t5.py`:
  `396313ebc2ba747f608e08697d78a324dc9c39baefeecbf8f8ca69562a1eb0a2`.
- `runs/confirmatory-manifest.json`:
  `4567ea4c07cae7e7f7cf1279c189eff94e02936aa77f28c8789ca48615871bab`.
- `runs/confirmation-data-hashes.json`:
  `044f24d31aebdd94074404a0a7c30440fcb4b8f0058a5d395b57c2f3b7c43e54`.
- Driver/watch, все14 config fields и data/config/phase hashes: как в base.
- Seeds20–37,18 парных единиц,54 runs, пилотный масштаб без гарантии80% мощности.
  Все thresholds, alpha.05, paired H1 contrast, все no-exclusion и integrity gates
  остаются прежними. Ни checkpoint, ни исход, ни объём не выбираются по fresh качеству.

## Единственное содержательное дополнение реализации

Новый `runs/analyze.py` SHA256
`e18de596ecb9abe250c6bcdcc85cc31b465b385810f50ca000b6dfd6a4e43bb6`
заменяет metadata-указатель base на старый analyzer6d479cc1…54dd3a.
Тесты `runs/test_analysis.py` SHA256
`003b88e86da1c12f5aa1e5a6a96273c463d4568cb8f965d2f31a747d7623564f`.

Для каждого из54 runs `feasibility.per_run` теперь также выводит:

- `C_dev_initial`, `C_dev_final`: accuracy из сохранённых phase2 step0 и step150.
- `C_dev_baselines`, `C_comp_baselines`: `n`, `majority`, `oracle_copy_A`,
  `oracle_copy_B` на точных frozen dev/comp records.

TargetC=max(field0,field1); majority=max label fraction. CopyA/B верен, когда
соответствующее истинное field0/field1 равно targetC. Ties могут засчитываться
обоим, поэтому сумма copy baselines не обязана быть1. Это аналитические baseline
решения, не дополнительные предсказания модели. Ручной пример в addendum даёт
majority.5,copyA.5,copyB.75. Все existing C-AUC/c_comp_pre/post/gain сохраняются.

`validate_h3_report` требует полный18×3 набор и обязательные поля до core summary
и публикации. Пропавший C_comp_baselines блокирует анализ, не исчезает молча.
Никаких новых forward/backward, training данных, статистических тестов или
cross-arm native контрастов. В power caveat исправлены только пробелы.

## Проверки и границы

Actual exact-source unittest:17 tests PASS, rc0. Ручные dev/comp baseline cases,
endpoints и наличие всех54 записей проверены; missing H3 field блокирует расчёт.
Три сломанных validator controls обнаружены3/3; fixtures только искусственные.
Actual admission --check по-прежнему блокируется отсутствием review gate, rc1;
output не создан. Обучающий код, manifest и datasets не менялись.

Independent spec reviewer: ACCEPT_H3_REPORT_DELTA_ONLY; independent17 tests PASS.
Claude Opus integration: APPROVE, 50s, exact H3 delta. Full review:
`<local>/review-20261023-152355.md`.
Его замечание о неизменном отсутствии self-SHA gate у analyzer закрывается
явным operator step: непосредственно перед inference оператор сверяет реальные
SHA analyzer/test/driver/watch/base/active/source/manifest/data с frozen panel
и gate receipt. Несовпадение блокирует inference и требует новой ревизии/ревью.
Это проверка оператором, не автоматическая гарантия неизменности analysis кода.

Дополнительная реконструкция ожидаемых H1 input bytes на3 dev+18 fresh seeds,
без модели/forward/training: внутри каждого seed они одинаковы между arms.
Это реконструкция из frozen source+records, не отсутствующий сохранённый live
H1 tensor hash. Blind spot и различие видов доказательства из base сохраняются.

## Приёмка и запуск

Первая exact-панель на base/старом analyzer дала Codex REQUEST_CHANGES по H3
report и Grok ACCEPT. Это НЕ два согласия на новую ревизию. Старые evidence
сохраняются в `reviews/freeze-v1-*`; все старые targets оставались неизменны.

После проверки дополнения эту ревизию требуется заморозить, получить новые
отдельные exact-target Codex и Grok ACCEPT. `gate.preregistration.path` указывает
на этот active v2 файл с его SHA, а не на superseded review target v1.
Receipt также фиксирует неизменный base SHA и новый analysis SHA. Перед созданием
gate повторно читаются все referenced bytes. До обоих ACCEPT fresh training закрыт.

Внешней регистрации/публикации нет. Данные пилотов0–2 не входят в fresh inference.
Все исходные ограничения, including floor-limited common H2 и within-arm-only
native diagnostics, остаются. Исходный вопрос о latent forgetting не закрыт.
