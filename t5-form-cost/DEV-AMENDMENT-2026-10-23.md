---
title: T5 dev amendment, native accuracy change
date: 2026-10-23
type: reference
stage: raw
status: PROPOSED_BEFORE_FRESH_DATA
tags: [machine-pedagogy, t5, pilot, retention]
---

# Operational native accuracy change, dev v2

Дата до первой правки v2: 23.10.2026. Первые девять dev0–2 уже просмотрены;
fresh seeds не запускались. [Статистика v1](pilot-summary.json), [PREREGISTRATION](PREREGISTRATION.md),
[README](README.md). Исходник v1 SHA256
`3da882c8cb99ee0513c9f98cae82a22ef7750c859a89d86b3069c41ac05a7a70`.
Все девять JSON, исходный manifest и source сохранены коммитом `333a7e8459`.

Причина: A-native-pre=1.0 во всех9, но A-common-pre у fixed/permuted
0.117–0.180, около majority baseline0.148–0.195. Common post-pre математически
измерим, однако не показывает, сколько освоенного A утрачено: общий экзамен
почти не выражает это умение ещё до B. Это не доказательство хорошей памяти.

Поправка ограничена прибором. Исходные обучение, данные, четвёртая грамматика,
H1 и H3 сохраняются. Новая пунктуация/сериализация по результату не подбирается.
Сразу после B добавляется один no-grad probe A в собственной учебной форме
каждого плеча. До B такой probe уже был `a_acquisition_native`.
Одинаковые A test objects, token/input/label bytes и horizon до/после проверяются
хэшем. Веса и RNG не меняются от измерения. Дополнительная цена каждого run:
256 примеров,3840 токенов,2 forward,0 backward.

Новая secondary колонка: `retention_native_delta = A_native_post - A_native_pre`,
operational native accuracy change. Только внутри каждого плеча; cross-arm
contrast, pooling и ранжирование native delta не выполняются. Формы плеч имеют
разную историю знакомства с порядками. Эта диагностика не заменяет исходный
common H2 и не закрывает неидентифицированный вопрос о потере освоенного common A.

Ни native, ни common probe не измеряет латентное уничтожение/сохранение навыка.
Изменение readout на этих входах совместимо и с interference, и с изменением
представления. Без no-B контрольной ветки нет выделенной причинной оценки вреда B.
Common H2 остаётся operational cross-serialization change с явными pre/post/delta.
Никакой суммы разных координат в индекс не строится.

## Приёмка v2 до её кодирования

1. Только дополнительный диагностический probe; training minibatches/C bytes,
   H1/H3 и все прежние метрики должны совпасть с v1 побитно при dev0–2.
   Любое расхождение вне новых полей/времени/счётчиков делает v2 run невалидным;
   новая колонка не интерпретируется до исправления причины отдельной ревизией.
2. Новые результаты в отдельном `runs/pilot-dev/native-retention-v2/`; исходные
   девять JSON и manifest неизменны. Повторно только0–2, ни одного fresh seed.
3. Общий исходный cap30мин обучения сохраняется: ledger v1 уже294.797сек,
   следовательно остаётся1505.203сек, включая новые импорты/неудачные попытки.
4. Парность native/common pre/post probe bytes; read-only веса/RNG; живой
   счётчик ожидает614400токенов,595forwards,550backwards на каждый полный run.
   Единственная новая категория `retention_native_post_test`: те же256 A test
   examples,3840tokens,2evalforward,0backward, сразу после B до C. Detached
   evaluation RNG; dropout0; actual weight hashes before/after равны, Torch RNG
   не меняется. Это запись в результате, а не только намерение теста.
5. Reviewer: независимый Codex для dev-кода; Claude для интеграционного diff.
   Exact-target Codex+Grok ACCEPT для frozen confirmation требуется отдельно.

## Проспективные ворота пригодности

Ворота служат для DEV и интерпретации; наблюдения fresh никогда не исключаются
постфактум из-за слабого освоения. Ни знак контраста, ни победитель воротами не являются.

- Native A-pre: >=0.90 в каждом dev run. Native B: mean>=0.80 каждого плеча;
  каждый отдельный run<0.80 явно показан, не исключён и без чтения как loss-after-acquired-B.
- H1: отсутствует общий пол, если хотя бы одно плечо имеет mean neutral B
  >=max(0.25, actual majority baseline +0.10); отсутствует общий потолок,
  если не все плечи имеют mean>=0.95. Это не гейт на превосходство variable.
- Native operational change содержательно интерпретируется только на run, где
  native A-pre>=0.90, native B>=0.80 и pre/post bytes равны. Низкий post может
  быть результатом, не основанием исключить seed. Это не cross-arm тест забывания.
- Common H2 интерпретируется только как change на общем экзамене. Утверждение
  об изменении освоенного common A допустимо лишь на run с его собственным
  pre>=0.80; иначе аспект floor-limited независимо от знака. Латентное forgetting
  не следует и при высоком baseline.
- H3: показать начальную/конечную dev accuracy, raw learning AUC и composition
  pre/post/gain с actual majority baseline и oracle-copy A/B baselines.
  Никаких ворот по ранжированию плеч.

Majority baseline = max доля одного истинного label в соответствующем уже
зафиксированном test/probe, не предсказание модели. Для dev0,1,2 соответственно:
A-test0.16015625/0.1484375/0.1953125; B-test0.15234375/0.16015625/0.14453125;
C-dev0.2265625/0.203125/0.234375; C-comp0.3125/0.328125/0.3515625.
H1 arm mean сравнивается с max(0.25,mean B-test majority+0.10)=0.25234375.
В v1 variable mean0.3606770833 проходит; общий пол не установлен, фиксированные
плечи близки к полу. Если v2 H1 не совпал с v1, это integrity failure, не новый эффект.
При общем провале frozen gate H1 помечается floor-censored, не научным нулём.

Первая внешняя панель проектного выбора: фактически ответили rails Codex и Muse,
Grok timeout300s, Gemini429. Оба ответа REVISE_DEV для исходного полного вектора;
Muse не объявляется другим вендором без доказанной фактической модели. Их замечание:
повышение common baseline до0.34 только позиционным shortcut не решило бы
чувствительность к забыванию. Именной Grok ответил REVISE_DEV_AMENDMENT: запретить
silent replacement common H2, cross-arm native contrast и чтение loss-after-acquired-B
при слабом B; добавить per-run gate, actual weight/RNG audit и disposition bitwise
mismatch. Эти замечания внесены до v2 rerun. Это не приёмка frozen protocol.
