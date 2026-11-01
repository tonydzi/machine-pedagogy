---
title: T5, completion of preregistered H3 reporting
date: 2026-10-23
type: reference
stage: raw
status: PROPOSED_BEFORE_FRESH
tags: [machine-pedagogy, t5, preregistration]
---

# Дополнить обещанный H3 отчёт, не менять обучение

Первый exact-target Codex review вернул REQUEST_CHANGES на frozen
[PREREGISTRATION](PREREGISTRATION.md) SHA d51a14d0c38348ce3c2e5234c2e50665ee3146b4c71216c92882336069f7ee48
и analyzer6d479cc1ecc9a57e8ac0da69227e821e744a5f921f5ecbcaea01924b3354dd3a.
Исходный протокол остаётся неизменным. Fresh models ещё не запускались.
Grok принял тот же первый комплект. Его ACCEPT не относится к новому analysis
target и active v2 protocol; обе исходные рецензии сохранены в reviews/freeze-v1-*.

Пробел: H3 в протоколе требует начальную/конечную C-dev accuracy и baselines,
но итоговый JSON переносит только metrics/wall, оставляя endpoints в raw curves,
а baselines в raw records. Это недостаток доклада, не отсутствие исходных данных.
Не выбираем новую метрику, исход, checkpoint или успешное плечо по результату.

Изменение ограничено `runs/analyze.py` и его тестами. В отчёте для всех54 runs
добавить C-dev accuracy на phase2 step0 и step150, C-dev/comp majority baseline,
oracle-copy A/B baselines на тех же dev/comp records. Сохранить existing C-AUC,
c_comp_pre/post/gain. Majority=max доля истинного labelC=max(field0,field1).
Oracle-copy A/B=доля объектов, где соответственно field0/field1 равен labelC.
При ties оба copy baselines могут быть верны; их сумма не обязана равняться1.

Контроль ручного счёта, только synthetic unit fixture:
records(0,3,0,0),(7,1,0,0),(2,2,0,0),(3,7,0,0) дают C labels3,7,2,7;
majority=.5, copyA=.5, copyB=.75. Тест требует правильные endpoints и наличие
отдельной записи для каждого из54 runs; native cross-arm contrast не появляется.
Дополнительных forward/backward, данных обучения или научных p-values нет.

Training source396313ebc2ba747f608e08697d78a324dc9c39baefeecbf8f8ca69562a1eb0a2,
manifest4567ea4c07cae7e7f7cf1279c189eff94e02936aa77f28c8789ca48615871bab,
data freeze044f24d31aebdd94074404a0a7c30440fcb4b8f0058a5d395b57c2f3b7c43e54
не меняются. Seeds20–37, n18, все thresholds, alpha, контраст H1 и no-exclusion
сохраняются. Исправление отчёта не требует нового dev training.

Приёмка до fresh: actual tests/negative controls, Claude integration review и
новые независимые exact-target Codex+Grok ACCEPT на active v2 protocol и новый
analysis SHA. В [PREREGISTRATION-v2](PREREGISTRATION-v2.md) будет thin revision со ссылкой на immutable
base d51; базовый протокол не переписывается. До этого gate receipt отсутствует.

Реализация дополнения: analyzer SHA256
`e18de596ecb9abe250c6bcdcc85cc31b465b385810f50ca000b6dfd6a4e43bb6`, tests SHA256
`003b88e86da1c12f5aa1e5a6a96273c463d4568cb8f965d2f31a747d7623564f`.
Actual exact-source17 tests PASS, rc0; три broken-validator controls обнаружены3/3.
Независимый spec reviewer выполнил17 tests, rc0, и принял только H3 report delta.
Это не заменяет integration review и обязательную новую exact-target панель.
