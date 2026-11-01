---
title: T5 form-cost, fresh-seed pilot-scale protocol
date: 2026-10-23
type: reference
stage: raw
status: FROZEN_BEFORE_FRESH_PENDING_EXACT_REVIEWS
tags: [machine-pedagogy, t5, preregistration]
---

# T5: стабильность формы и вектор цены обучения

**Протокол заморожен до fresh обучения: 2026-10-23 13:56:17 UTC.** Pilot-dev v1 завершён
9/9. Девять результатов и его source сохранены Git333a7e8459. Из-за низкого A-common
baseline добавлено диагностическое измерение в отдельной dev-v2 ревизии.
Точный scope поправки: [DEV-AMENDMENT-2026-10-23](DEV-AMENDMENT-2026-10-23.md), наблюдения: [pilot-analysis](pilot-analysis.md).
Dev parity и вход final analysis проверены. Независимые exact-target Codex+Grok
ACCEPT требуются отдельно; до них fresh обучение закрыто. Этот файл после freeze
не редактируется; последующий статус/receipts находятся в README и runs gate.
Внешней регистрации и публикации нет. Источник: prereg-draft-encoding-ergonomics (not exported).
Приёмка: task (not exported). Статус: [README](README.md).

## Вопрос и границы

Двухсторонний вопрос: как стабильность шаблона меняет новое качество, сохранение
старого навыка, последующую обучаемость и полную вычислительную цену?
Направление эффекта не заявляется; новизна не установлена. Корпус Дзятковской
мотивирует вопрос и совместный учёт цены. Опыт не изолирует психологический
механизм книги, не переносит человеческие проценты на нейросеть, не измеряет welfare.
T4 меняет порядок/отбор/replay, T5 меняет форму внутри примера. T3 и GPU исключены.

## Данные и три навыка

Запись: четыре именованных поля со значениями 0–7. Атомарный словарь 18 token IDs,
без BPE. Ответ из восьми классов хранится отдельно от input. Навык A извлекает
первое семантическое поле; B второе; C вычисляет их максимум.

Train/dev/test разделены по полному tuple глобально между всеми навыками.
Для переноса C отдельно исключены восемь unordered пар первых двух значений,
в обеих ориентациях: {0,3}, {1,4}, {2,5}, {3,6}, {4,7}, {0,7}, {1,6}, {2,7}.
Все значения в них неравны: простое копирование A либо B не решает весь контроль.
До обучения C измеряется `c_comp_pre`, после него `c_comp`, разность `c_comp_gain`.
Это перенос на новые пары значений, а не доказательство общего композиционного умения.

Генератор перечисляет 4096 tuples; 1024 composition, 3072 остальных. Остальные
перемешиваются seed-specific stream и делятся 2048/768/256. Для каждой фазы
независимая выборка 512 train, 128 dev, 256 test, 128 composition.
Раздельные RNG: model init, global partitions, phase sampling, minibatches,
phase-specific form stream и deterministic evaluation. Dropout=0.
Порядок семантических minibatches совпадает между плечами; реальный порядок
логируется хэшами, а реальные входы C имеют отдельный hash для парной проверки.

## Три плеча и общий экзамен

| Плечо | Учебные фазы A/B |
|---|---|
| fixed | Постоянный порядок [0,1,2,3] блоков key,value,SEP |
| permuted | Другой постоянный порядок [2,0,3,1] тех же блоков |
| variable | Независимая перестановка блоков при каждом предъявлении |

Во всех плечах совпадают смысл, ответы, token multiset, длина 15, позиция маркера
ответа и loss mask. Каждый реальный batch проверяет round-trip и позицию loss.
Uniform instruction token и task marker одинаковы. Никаких gold answer tokens во входе.

Общая четвёртая грамматика: value,key,SEP в каждом блоке, с arm-independent
перемешиванием порядка блоков. В A/B её не обучают. Она отличается от permutation
целых блоков; словарь полностью известен из обучения. Все checkpoint probes
используют одинаковые входы, no_grad и не меняют оптимизатор.

**Асимметрия H1 явная:** variable уже видело распределение порядков полей,
а fixed/permuted сталкиваются с новой смесью порядков вместе с обратной парой.
H1 измеряет перенос стабильности/вариативности вместе с аугментацией и этой
асимметрией. Он не изолирует эргономику или универсально лучшую форму.

Фаза C идёт после фиксации H1/H2: одна и та же четвёртая грамматика, реальные
входные bytes, record order и budget во всех плечах. Это новый навык после A/B.

Контрбалансировка: `seed%3` задаёт позиции ANSWER 2/8/14; `(seed//3)%3`
задаёт три циклических mapping ключей. Внутри seed одинаково по всем плечам.
Модификация контраста по этим блокам только описательная, без новых primary tests.

## Конфигурация, после dev выбора и до fresh результатов

| Параметр | Значение pilot-dev |
|---|---|
| Сеть | CPU Transformer encoder, 2 layers, d80, 4 heads, FF160, pre-norm, dropout0 |
| Параметры | 107448, измерено selftest |
| Оптимизатор | AdamW, lr0.001, weight_decay0 |
| Threads / batch | 1 / 64 |
| Шаги A/B/C | 200 / 200 / 150 |
| Checkpoints | 0, каждые25, последний шаг |
| Библиотеки | Python3.12; torch2.14.1+cpu; numpy exact version в JSON run |
| Replay / restoration | 0, одинаково во всех плечах |
| Pilot seeds | 0,1,2, никогда не входят в confirmation |
| Candidate source SHA256 | `3da882c8cb99ee0513c9f98cae82a22ef7750c859a89d86b3069c41ac05a7a70` |
| Native-probe v2 source SHA256 | `396313ebc2ba747f608e08697d78a324dc9c39baefeecbf8f8ca69562a1eb0a2` |
| Spec SHA256 | `4dd62b90f50f922af4b582bfa44856a2c31e0db148c08f07182757a05595935f` |

Fresh серия конечная: seeds20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37.
18 парных единиц,54 runs; каждая из девяти mapping/answer-position cells повторяется
дважды. Это новая серия **пилотного масштаба**, с заранее заданным анализом и
конечным объёмом. Название `confirmatory` у driver обозначает технический гейт
до fresh данных, не обещает мощности. Пилотные0–2 не включаются и не объединяются.

Проектный порог практического интереса3п.п. не является ожидаемым эффектом.
V1 H1 seed contrasts=.21875/.26953125/.1953125, SD=.0379395893 при n3.
Для n10/n13/n18 noncentral-t power при delta=.03 и ровно этой SD:
0.6063/0.7446/0.8850. Вычисление: tcrit=t(.975,n−1), ncp=.03/SD*sqrt(n),
power=P(T_nct>tcrit)+P(T_nct<−tcrit). Условная мощность не учитывает огромную
неопределённость SD из3 units и новые mapping. Поэтому **нет гарантии80% мощности**.
Объём выбран для покрытия/конечной точности, не по размеру наблюдавшегося эффекта.
Ни увеличения n, ни optional stopping, ни выбора сидов/checkpoints после fresh нет.

V1 runtime: internal sum270.9435403001262s, per-run mean30.10484s, range19.8363–73.7042;
driver elapsed294.7970000002533s. Весь dev budget1800s остаётся общим для v1+v2.
Wall outlier/preprocessing и постоянный порядок исполнения не дают причинного
сравнения скорости форм. Энергия/FLOPs не измерены. Точные versions в результате:
Python3.12.10, torch2.14.1+cpu, numpy2.5.3; analysis scipy1.18.1.

## Точные frozen inputs, подготовлены без fresh обучения

- `runs/confirmatory-manifest.json` SHA256
  `4567ea4c07cae7e7f7cf1279c189eff94e02936aa77f28c8789ca48615871bab`.
- `runs/confirmation-data-hashes.json` SHA256
  `044f24d31aebdd94074404a0a7c30440fcb4b8f0058a5d395b57c2f3b7c43e54`.
  Внутри exact config digest, layout и data_manifest/split/phase-split hashes
  по каждому из18 seed. Генерировались только данные, без модели, forward или training.
- Полные14 config fields в обоих manifests: steps_a200,steps_b200,steps_c150,
  threads1,batch64,width80,layers2,heads4,lr.001,checkpoint25,
  train_n512,dev_n128,test_n256,comp_n128. Source SHA указан в таблице выше.
- Driver SHA256 `dfae68b767d9db861af1dffd1dddb01d963758b8f41746cd012886c7e42c989d`;
  watch SHA256 `a0762ec5d90124c6a99097213b6bd6e14f7566b4eafe40bc49e80a4c10d0cbfe`.
- Analysis entry `runs/analyze.py` SHA256
  `6d479cc1ecc9a57e8ac0da69227e821e744a5f921f5ecbcaea01924b3354dd3a`;
  tests `runs/test_analysis.py` SHA256
  `f4e0fdc850632d190f3a10d574ce468524be3a33319b304e2b4931b927f049c3`.

Fresh labels и hashes не использовались для выбора обучения, grammar, gates или n.
Полные actual data_manifest hashes из54 результатов должны совпасть с hash-only
freeze. Настройки хэшей/драйвера не заменяют проверку реальных tensor bytes.

Вход в final analysis: сначала `driver.py --manifest runs/confirmatory-manifest.json
--check` обязан вернуть JSON `valid=true,complete=true`; rc0 без complete не достаточно.
Конкретный анализный вход: `python -B runs/analyze.py --manifest <absolute
runs/confirmatory-manifest.json> --out <absolute final-summary.json> --check`.
После его exit0 снять `--check` для единственного итогового JSON. `--check`
не импортирует модель/статистику и не пишет. Wrapper в inference вызывает core
summarize только после всех54 проверок; прямой запуск general summarize не admission.
Затем перечитать ровно54 expected пути из18seeds×3arms, проверить точное множество
идентичностей, каталог содержит только эти54 regular result files (никаких иных
файлов/подкаталогов, включая logs/tmp), и соответствие каждого
data_manifest_sha256 его seed в hash-only freeze. Проверить code/config/manifest
SHA и actual semantic/C-input/probe/budget invariants каждого run.
Только после этого вызвать `t5.py summarize --files <точные54пути>` и потребовать
`n_paired_seeds=18`. Сам summarize принимает произвольное n и не является gate;
передача17 полных triplets без этих проверок нарушает протокол и BLOCKS inference.

Граница прямой телеметрии: source396 сохраняет actual common/native A probe hashes,
actual common dev-curve input hashes всех25 checkpoints и actual C-training inputs.
Отдельный actual H1 B-test input hash не сохранён. Его одинаковость определяется
точным source, одинаковыми frozen B-test records и независимо заново созданным
neutral evaluation RNG, а не сверкой отдельного сохранённого H1 tensor hash.
Не приписываем отсутствующую телеметрию задним числом.

## Метрики и статистика

H1 primary: accuracy B после фазы B на общей четвёртой форме; один парный контраст
на seed: variable minus mean(fixed,permuted). Двухсторонний one-sample t-test
парных разностей, alpha0.05, 95% t-CI. Единица анализа seed, все разности показываются.
Широкий нулевой CI означает неопределённость, не эквивалентность.
При SD парных разностей=0 t-test/t-CI не определяются и остаются null;
mean и все18 разностей показываются, эквивалентность не объявляется.

H2 secondary original: accuracy A в четвёртой форме непосредственно перед/после B;
retention_delta=post-pre. Показываются все pre/post/delta. Это operational common
probe change. При собственном A-common-pre<.80 run получает floor-limited и не
интерпретируется как изменение уже освоенного common A. В v1 все9 ниже этого порога.

Добавленная secondary диагностика: A-native-post−A-native-pre на тех же256
heldout A examples и тех же actual x/at/y bytes, после B перед C. Только within-arm;
никаких cross-arm native contrast, pooling или rank. Содержательная интерпретация
данного run только если A-native-pre>=.90 и B-native>=.80 и bytes равны. Слабые
наблюдения остаются без исключения. Это operational accuracy change, не latent
forgetting, не замена common H2 и не причинный ущерб B без no-B control.

H3 secondary: нормированная трапециями AUC dev learning C от0 доstepsC;
начальная/конечная dev accuracy, accuracy/gain на новых парах C до/после обучения.
Majority baseline=max class-label fraction на данном наборе; oracle-copy A/B
baseline=доля объектов, где targetA/targetB уже совпадает с max. Положительный
gain сам по себе не доказывает общее композиционное умение.
Матрица B на трёх учебных формах вторична. Никакой замены H1 удобным checkpoint.
H2/H3 и бюджет описательные; дополнительных p-значений не объявляется.

Бюджетный вектор: train/eval/probe tokens, padding0, actual forward/backward
calls, preprocessing wall time, CPU process seconds, полный wall time.
Независимый forward-hook сверяет реальные вызовы с ledger. FLOPs и энергия
не измеряются. Steps не называются FLOPs; координаты цены не складываются в
произвольный «индекс здоровья». Межплечевое Pareto-сопоставление ограничено
сравнимыми общими H1/H3/resources. Native delta остаётся within-arm диагностикой,
не координатой межплечевого ранжирования. Common H2 несёт собственный floor flag;
из него не выводится латентная сохранность.

## Инструмент, отказы и ворота подтверждения

До обучения: selftest и red_proof на копиях для token drift, answer leakage,
неверного binding, semantic duplicate, скрытого forward, placebo gradients,
counterbalance, испорченной retention/C-AUC и тривиального composition holdout.
Код не меняется внутри run; начальный/финальный source SHA должны совпасть.

Технический `status=complete` не означает научную валидность. До fresh приняты
численные правила пригодности и интерпретации:

- A-native-pre>=.90 для каждого run; B-native armmean>=.80, каждое значение<.80
  отмечается. Это feasibility flag, ни один seed не исключается из H1 анализа.
- Общий floor H1 отсутствует, если хотя бы одно armmean B-common
  >=max(.25,mean actual B-test majority+.10). Общий ceiling, если все armmeans>=.95.
  Правило не требует превосходства конкретного arm и не выбирает знак контраста.
- Whole-assay `assay_limited = any(A-native-pre<.90) OR
  any_arm(mean B-native<.80) OR H1_global_floor OR H1_global_ceiling`.
  Низкий B-native отдельного run отмечается в native diagnostics, но не меняет
  этот Boolean сверх его armmean. Common H2 floor flag не является H1 floor.
- При assay_limited или нарушении parity вывод
  ограничивается assay/feasibility. Формальный предзаданный H1 estimate/CI/p
  сохраняется с флагом, а не выдаётся за научный нуль/эффект. Все18 units показаны.
- Native/common H2 per-run gates и запреты изложены выше. Малый post не причина
  исключить run; native delta не агрегируется между arm.
- В v2 live проверяются actual native weights before=after и Torch RNG unchanged;
  новые probe3840tokens/256examples/2evalforward/0backward. Ожидаемый полный
  budget614400tokens/40960examples/595forward/550backward во всех54 runs.
- Actual semantic batch hashes совпадают внутри seed; actual C-input tensor hashes
  совпадают внутри seed; paired common/native pre/post input+label hashes равны.
  При нарушении integrity сохраняются JSON/логи, серию не интерпретируют как эффект.

Все54 observations сохраняются. Инфраструктурный неполный run разрешено повторить
максимум3 раза по заранее заданному driver; completed result не перезаписывается.
Неполный paired seed не включается молча: inferential series BLOCKED, требуются
причина и отдельный dated amendment/review, не post-hoc n17. Низкое качество
полностью завершённого run не является технической ошибкой и не перезапускается.

Windows driver/windowless watch: точный manifest, SHA/config/seed/arm readback;
три CPU samples<=70%, один поток, низкий приоритет; cap1800s суммарного pilot
обучения и максимум3 попытки. Логи/ledger локально, данные в Vault. Watch считает
только python.exe workers; driver защищён Scheduler IgnoreNew и OS lock.
Completion подтверждается полными валидными JSON, не строкой ALL DONE.
Инфраструктурный отказ классифицируется по логу до рассмотрения качества.

**Подтверждение закрыто:** требуется frozen source+data manifest+config+seeds,
независимые exact-target Codex и Grok ACCEPT, хэши полных review evidence и
PREREGISTRATION. Дополнительный интеграционный reviewer Claude. До этого никакого
fresh-seed run. Поля ACCEPT или measured runtime здесь не заполняются намерениями.

## Status log и изменения до данных

- 23.10: документ создан как DRAFT_PENDING_PILOT_NOT_FROZEN, результатов ещё нет.
- До первого обучения: независимое ревью нашло общий RNG формы C и тривиальный
  equal-pair holdout max. Исправлены; старый код не породил научных результатов.
- До первого обучения: исправлены config contract, hidden-forward accounting,
  смешение manifests в summarize, orphan/timeout/test-launch checks драйвера.
- 12:12 и12:17 UTC CPU-гейты отложили обучение. Это запуск драйвера, не результат опыта.
- 12:36 UTC v1 завершён9/9: см. [pilot-analysis](pilot-analysis.md). Из-за floor исходного common H2
  подготовлена dated dev-v2 amendment; H1/H3/training/grammar не менялись.
- 13:16 UTC v2 завершён9/9; исходные метрики и scientific trajectories побитно
  совпали с v1. Actual semantic/C-input hashes парны; actual probe bytes парны;
  weights/RNG audit pass9/9; +3840tokens/+2forward без training. v2 driver151.122s,
  cumulative v1+v2=445.919s<1800. Source+results сохранены backupac696cc13b.
- Новый native A post-B=.15234375/.16015625/.20703125 для fixed и permuted
  по seeds0/1/2; variable=.1640625/.16015625/.30078125. Это observed operational
  performance change от A-pre1.0, не latent forgetting. Variableseed2 B=.69921875
  flag сохранён, observation не исключается. Common H2 по-прежнему floor-limited.
- До freeze: analysis entry15testsPASS, rc0, exact SHA выше; independent Codex
  ACCEPT_ANALYSIS_ENTRY_ONLY, Claude integration APPROVE receipt
  `<local>/review-20261023-145243.md`.
  Три сломанных validator controls обнаружены3/3 на этом exact source.
  Real --check rc1 на отсутствующем gate, inference/output отсутствуют. На реальном
  E: filesystem в _scratch synthetic publish создал новый файл, повтор использовал
  те же bytes, попытка другой записи отказала; raw experiment data не использовались.
- Freeze 2026-10-23 13:56:17 UTC: source/data/config/seed/analysis targets фиксированы.
  Полные15test stdout и real-refusal в
  `<local>/analysis-v3-live-evidence.json`.
  Final gate создаётся только после отдельных Codex/Grok ACCEPT на SHA этого файла.
