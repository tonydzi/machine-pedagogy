# Sanitization of this export

The experiment ran in a private working folder. Before publication every file in this
folder (except the hand-written `README.md`, this file and `tools/`) was produced from
that folder by [`tools/sanitize_export.py`](tools/sanitize_export.py), the exact script
used, re-runnable as

```
python tools/sanitize_export.py SOURCE_DIR TARGET_DIR \
    --redact "<producer machine name>=producer-node" \
    --redact "<private runtime folder name>=T5Agents"
```

The two private literals are passed on the command line so that the script does not
republish what it removes.

Sanitization changes file bytes, so every hash frozen in the preregistration, the gate
receipt and the analysis provenance refers to the **original** bytes. The table below
maps each original sha256 to the exported sha256, so a frozen hash can be traced to the
file published here.

## What was changed

1. Provenance stamp lines removed: lines starting with `Machine:` (internal machine,
   agent and person stamp) in Markdown and in Python docstrings.
2. YAML frontmatter keys for machine, operator, author and origin removed; `title`,
   `date` and the other keys kept.
3. The internal producer machine name replaced by the neutral label `producer-node`
   wherever it is data: the `MACHINE` constant in `t5.py`, the event `node` in
   `runs/driver.py`, and the `Machine` / `usage.node` / `data_manifest.Machine` values in
   every JSON file. JSON members naming the human operator removed.
4. Absolute local paths replaced: paths under the experiment folder became paths relative
   to this folder (for example `runs/confirmatory-manifest.json`); the Python interpreter
   path became `python`; private scratch, runtime and review-log paths became
   `<local>/<file name>`.
5. Internal run codes removed: `T5-form-cost-<code>` became `T5-form-cost`, the runtime
   folder name became `T5Agents/t5-form-cost`, and a bare code in prose became "this run".
6. Private notebook wiki links (`[[...]]`) became relative Markdown links when the target is in this export,
   and plain text (marked "not exported") when it is not.

Not changed: every number, seed, statistic, hash string, metric, curve, ledger entry and
data record. JSON files were edited at text level (formatting and CRLF line endings kept)
and the script re-parses each one and asserts that it equals an object-level transform of
the original restricted to the fields above. Code changes are limited to the constants in
item 3 and the names in item 5; no logic was edited.

## Consequences for verification

- `t5.py` differs from the frozen core (`code_sha256` 396313eb...). The original hash
  is the first row of the table.
- `data_manifest.Machine` is part of the hashed data manifest, so recomputing
  `data_manifest_sha256` from an exported run JSON does **not** reproduce the stored
  value; the split, phase, probe, minibatch and C-input hashes do not include the label
  and still verify. A fresh `python t5.py run` writes `producer-node` into its own data
  manifest, so its `data_manifest_sha256` also differs from the frozen value while
  `split_sha256` and `training_semantic_sha256` match (checked on pilot seed 0, variable:
  identical split and minibatch hashes, identical H1 accuracy 0.35546875).
- The strict admission chain (`runs/driver.py --check`, `runs/analyze.py`) requires
  absolute paths and the original raw manifest, code and result bytes. On this export it
  stops at the first check by design (`absolute path required`). Use
  `python t5.py summarize` for the statistics and this table for byte provenance.
- Not exported: internal review transcripts and receipts (`reviews/`), the task note,
  the private status README, HTML status screens, the scheduler and watchdog scripts
  (`runs/watch.py`, `runs/install_tasks.ps1`, `runs/test_driver.py`) and logs of local
  infrastructure. Files that reference them keep their recorded hashes.

## Hash table

New files, no original: `README.md` `837a222d6e31edf038d1e297311072c4da51ae19cf03e002f78da0920e67397b`, `tools/sanitize_export.py`
`fd72b11aa8614cb337ea26827216556ed37ac00167f6e43a7af70d29f33bd004`. This file is not hashed in itself.

| File | Bytes | original_sha256 | exported_sha256 |
|---|---:|---|---|
| `t5.py` | 27741 | `396313ebc2ba747f608e08697d78a324dc9c39baefeecbf8f8ca69562a1eb0a2` | `62312e1ba25fb97630ebd9b6c75bb1fbc784b45fd69e9fd6cab1e0f39aa3c3fe` |
| `PREREGISTRATION.md` | 24667 | `cc587bd406df50bb3c91a70634338c8f12ce700385827bfa698ba544b9884541` | `e30e69b704409c6950c27a7de9031f07538ba8edb78573112979090414a9c23c` |
| `PREREGISTRATION-v2.md` | 6802 | `0b29c716300e639e1aa91ef6a15bad501cf56565d0740abddeac58a1c89e73d8` | `1c91cfac352361eec13c99cc75c1abc41709b4bd2aecae62c99f6de30e4832ff` |
| `DEV-AMENDMENT-2026-10-23.md` | 8928 | `c0a21668901592410e546072c49a676eebc75a796760c539f999cf738fb6b4a1` | `4327fc33bcb9a49c33f9147cee6ba5d2a6affc416c41375c9db042620f120e66` |
| `ANALYSIS-ADDENDUM-2026-10-23.md` | 4003 | `00b66bcdc3fc97098388bbff048fe9db82b619ef1601460adea45db78a2f4b75` | `81ba5dd5d55b74b1a569bad97ac152a8c4661d5f9190ae9679add6441eae134d` |
| `RESULTS.md` | 15264 | `64287fc29b54faaa017aa84e6be6b4de53c9d07cc26e5a21d1f3ba055466478f` | `62227dac6ae17f75ce383a56bf26dc9c50533cc2c4a1a7f618f320513568c98b` |
| `pilot-analysis.md` | 7170 | `9d560dc5ff726cf4aeda1b7e7783f7bb6d68cf52c420148b982162f2276265b8` | `db2a61f5aeffbc2044a3bd41da0476858abef48fbd2e953df09a9b416415578e` |
| `final-summary.json` | 71281 | `836b43ef3f0d45a21b3ddece68e0d27f004ec1a9b545255c9f43060e3bd45a79` | `0b181b85236b23967b3f9be2ee1cb23a722360f620364467edec21f6b36ea974` |
| `pilot-summary.json` | 4522 | `47d168c33c6019b35f2f8f140f3a72cc224c50c067237755cd98e6d4a32200f5` | `68a7053c8220347a133da312777c7570e640a218f134cbf00c06fb39528c2d27` |
| `runs/analyze.py` | 18261 | `e18de596ecb9abe250c6bcdcc85cc31b465b385810f50ca000b6dfd6a4e43bb6` | `73d3dbc3b213acf1bcdcf1297feb7ce4289535dc2eeb9683c20f6d0bfcad7d58` |
| `runs/driver.py` | 18758 | `dfae68b767d9db861af1dffd1dddb01d963758b8f41746cd012886c7e42c989d` | `1ad3b8201cb680e42a561f5543a3237ee10ef80a50ddab97c203fc9ddfa43c53` |
| `runs/test_analysis.py` | 21853 | `003b88e86da1c12f5aa1e5a6a96273c463d4568cb8f965d2f31a747d7623564f` | `f429f2871bf58ba6ad57f9bc71bbf2e590248e2f80c24f935b631514ce722832` |
| `runs/confirmatory-manifest.json` | 1248 | `4567ea4c07cae7e7f7cf1279c189eff94e02936aa77f28c8789ca48615871bab` | `1d025a9e8e83bf7d5f21b76a36ddd7d40364d6b4e8cf1c59905ba00bb4b14e89` |
| `runs/pilot-manifest.json` | 776 | `baf2500e0b8fe32bbc4b74872b0eb3f1fde571ac5b465d11111b02cc93e77142` | `f18119bb6db505f74c0a3056923a9ebca5030aad94644e0e611eb839a80a2c19` |
| `runs/pilot-v2-manifest.json` | 961 | `b9f505be4d962519bf5f35f1c2ef2662a2476481945b16884b4118a730d766a1` | `a19e9e88a5b960ac4481d39c16f8ca387b50180bde262a6266ec235b5dd76426` |
| `runs/confirmatory-gate.json` | 2615 | `8c9a68ecaf9feedc21532b7bf02e89d4b20e7a181c8a475b1d5f0a90191fbe1b` | `abf28c6ea424159b436ac6547e598c82262ff33aa09e08c85e1f16d14d444ebd` |
| `runs/confirmation-data-hashes.json` | 40605 | `044f24d31aebdd94074404a0a7c30440fcb4b8f0058a5d395b57c2f3b7c43e54` | `09f6233af4a5e0a9f15d83d9bfc446465ab44426ac1eef8f408a06986ebfa41d` |
| `runs/confirmation-independent.json` | 10731 | `fc05e1627876ddf6e12c7cbcd741564a248458eee65e042806eb4447f177aa6c` | `4b3ca60f57278fdccfc692a3a5bb30064d1aa2ae7995b83163f26f053f495654` |
| `runs/confirmatory/native-retention-v2/seed20_fixed.json` | 290170 | `337c23e55e4ad72f24c0d9b17b73088614be8720cdde29caa2005f0a74afd74e` | `43ddc7ce125e02d33e20df77534766c1358badfe2e4212ff7c3df5cc76b03a23` |
| `runs/confirmatory/native-retention-v2/seed20_permuted.json` | 290152 | `c6e6079ae5671e422a863938a94dc36fbc017d1d6eb2dd8cb4ad83d18cbf9fd5` | `3d9e2296bc4c5e3d2c5e9490de0625d4b77deaef68944d6ab6edbd891b461a94` |
| `runs/confirmatory/native-retention-v2/seed20_variable.json` | 290200 | `63c68a38e66d52cd125129dc99527a5e474dc496f530641ee01c32ef7f8edc53` | `d7dda88013f7e1d77d341c4756bc2abc6619ead9ee03274817c0b26bb29e0905` |
| `runs/confirmatory/native-retention-v2/seed21_fixed.json` | 290184 | `06353d49d8476b7ac33391b12f4da49f76e10b36daee1f68dbef72f856f59c34` | `a5a02fc2902d7f43aad4745c6e4a237e4294ac03acba7934741699ba22276a46` |
| `runs/confirmatory/native-retention-v2/seed21_permuted.json` | 290161 | `af5a1a4c2182518f0408f4aedace1699e26aa6dd70b7caf0946ccdf4b912b98c` | `c3e5c127760b0ad00ec47a20252925709a04eb6694f5f56a36388cfee38f279d` |
| `runs/confirmatory/native-retention-v2/seed21_variable.json` | 290148 | `9944ae89a9520c4866c159c2e006f54a9246e2340a0c5781f3c5062f2cff88c2` | `8b0d0b8dcef70b623f3440ccb39d8bf372356c94617e408dc88c61c9eb4c966b` |
| `runs/confirmatory/native-retention-v2/seed22_fixed.json` | 290168 | `e2d810e8ca6118de3c3c779ebd2fb19bcd68da373522b01acba7762a3d9f07b0` | `9a8d55786bbbf4d7c56a7e89aeadb6a0c752b53278132edc120e2b54b29e18ea` |
| `runs/confirmatory/native-retention-v2/seed22_permuted.json` | 290200 | `7c52acc5fb2b69d66dfe05295220a5ee86f3da9fcb60588270bb1369496de68c` | `6806bbf0a694d12e5120dec1d0f008d5e6d50ecd973a419d8182e812893986dc` |
| `runs/confirmatory/native-retention-v2/seed22_variable.json` | 290167 | `aa9f2c97058d538a4d3b992dc1118955cf1d3eb68d835723379b7e99c6e9a08e` | `82f691f8c3091942badeee6239af36c1a2d084c562ddc63945a41037542638bc` |
| `runs/confirmatory/native-retention-v2/seed23_fixed.json` | 290179 | `af8edbd96dbe8676d93b1263cbf7283d401f5d62a0e28f868bdc58a27d06bb6e` | `2cfe2726c748113e12d2e00c41628e2522c1ce8730479c84d2995689b928f1ae` |
| `runs/confirmatory/native-retention-v2/seed23_permuted.json` | 290173 | `054f6b978c1c1c43f090992c00d5f5cb7ed6e768bcd90f5885aaf0bf42f5cfd2` | `e03691ccb612378ebce8ff16805ee864eb31f16aa5a723b8c691cbbae041b5f4` |
| `runs/confirmatory/native-retention-v2/seed23_variable.json` | 290174 | `957f9c24c6b6b3783870b2b3b6a2381f42bc4315b1bb090378a9a25f2e2bbd64` | `9be0ba6362f196bd0dbd5a4359bf5f9541f2c2cfb25fd894c240733e82ad49ce` |
| `runs/confirmatory/native-retention-v2/seed24_fixed.json` | 290146 | `5cb1f21272202747978322d061d8669bd5470cba199add5bcff545cb82dcfbd2` | `cc78a1fa88be6afac4c63b8bcacd2369413c5c929850def7a65df6610e39161d` |
| `runs/confirmatory/native-retention-v2/seed24_permuted.json` | 290199 | `4008e672b36b5727cded24282439e23d16cbb32bfe79b306e5eae871bbf6bb37` | `ce0c733dedd6efc47e9e829936669a13ca98ff59ce9e899fc037b812db28e475` |
| `runs/confirmatory/native-retention-v2/seed24_variable.json` | 290168 | `7d818a2e4df9522f999081af41b6180dfa9c25ef24275c63480021e5d894e44d` | `b98d6c417d73f9b89eecd1657ae4317578829aebb101d3407fc9c3254fca9bc2` |
| `runs/confirmatory/native-retention-v2/seed25_fixed.json` | 290173 | `fc6fc3a05b9bd83c9e5be9809d73c2262d12a6a4c2f750de440f124df0090d18` | `ac659c35295ee1bf7788a0900417bc5b4fa1c48bae5b04496272c5ddabb8b982` |
| `runs/confirmatory/native-retention-v2/seed25_permuted.json` | 290170 | `fc8a0cf90cf214f00b3c64098b2b5dc439ff192e217328b1218409abceace8b0` | `64aaebc22be6be9c322d0e460737cbad154eb052d4efa55d56b6c88c5f6ae6b6` |
| `runs/confirmatory/native-retention-v2/seed25_variable.json` | 290179 | `b388441729e43c2a64af22231afed6268098cf02bc3b3f5796284c67e07655f4` | `b0f4c0555ff7ddac72810ab958b067a772073581c8223c4d88ca4eb080ddeb25` |
| `runs/confirmatory/native-retention-v2/seed26_fixed.json` | 290149 | `e5074e33b3f6d3990445e5d9435967c33ac919759b75e000d76d011663826c1e` | `ec7b081d8528cc68e9f125e0b5b02c8821930f191f1983b69c92c3382bdb9fe7` |
| `runs/confirmatory/native-retention-v2/seed26_permuted.json` | 290168 | `738c3e1829a2c6181386838548e95249c0d10b273367ad778bb035f7d84985cf` | `97a07f8aa5f48e985d9ea2d1a5fb4c0b8050c0b69a8dc8b51c4c45ab1b2e1aea` |
| `runs/confirmatory/native-retention-v2/seed26_variable.json` | 290189 | `29e8a9eb5308c96cc6916fe682f6fb4e0c6032b4ff9f4b31d86656d70ab3a99f` | `3c7c7577694fedf64861f79249d1e1112633893f0620e7698a8229172e2f977b` |
| `runs/confirmatory/native-retention-v2/seed27_fixed.json` | 290168 | `6a0bb80a3d0c0e0f7a38b84f3dd648db4193915fd9bfbf14a2a193910a3106eb` | `b75b99f7869f5f72c093dcb73c5f89962bb68de0231d0c2df8cfb1fe1ca659fd` |
| `runs/confirmatory/native-retention-v2/seed27_permuted.json` | 290199 | `626bd95d7a4969ec64b13913fe581028e7e8907c00106a5a7dba56e25566c91c` | `cb8396003190749c83e3f709081bc836dc95326067d2a74230b7d4d2d968b350` |
| `runs/confirmatory/native-retention-v2/seed27_variable.json` | 290180 | `9efa6edf85ee313105c56f1112ede19fd9f4bff90cdf7cbfd1b5f943e7c754da` | `610ce2223490f122e52fd46c029d3f02a98ec734486bd2b0f2b91676a1bb293a` |
| `runs/confirmatory/native-retention-v2/seed28_fixed.json` | 290192 | `5864fb79882a28dc4cf6c568c04b25dceb8c734dfcdf156efafef580f6f76da9` | `4fb470a326eec20a9a9d82f76a71f1631ab6319f6e6bc8160e78540427fa6dbc` |
| `runs/confirmatory/native-retention-v2/seed28_permuted.json` | 290192 | `bf4bbc12fa6c1d2d4fc9dd3a31b488341de2d31e0f29c0cee6657962770f15d6` | `c9c04b1701ef987006a1b55b259088aed75f7ca46567fa1a8395212e82e53c33` |
| `runs/confirmatory/native-retention-v2/seed28_variable.json` | 290201 | `81e5f0cecb3ad1c9c36fa0523ee762c6b57a21038200b0fd903fd51f20f48c3d` | `09d0856a8ebeccc63adb1de4d385897568b47b1ab39961b4ad9f28d7862f3603` |
| `runs/confirmatory/native-retention-v2/seed29_fixed.json` | 290146 | `2918cf15a2ce335a029aeeb183e6e14d5615aa48431aaf4104570b619d48bd8c` | `1bca3d19084c12d57b03eb423c0a8b41ecec468de412cf272b3072539db5b1ba` |
| `runs/confirmatory/native-retention-v2/seed29_permuted.json` | 290188 | `f2f4f912f56e24f0847b0ec8ab6ffc64b30aaf0aa0534dece18a7e9d5f0811cc` | `86a9d7a548837d631f02f820198dbf9b15ba193aff5e2d3170e1bb6f36d7ea0e` |
| `runs/confirmatory/native-retention-v2/seed29_variable.json` | 290148 | `e386941ca446148dd12381cdecab6e655c275fad16d2aa8976944d677b9a0f80` | `75c0caac26437eea7e0969bae2fb6a27242aea747156a70c53e0df3c3a31c60c` |
| `runs/confirmatory/native-retention-v2/seed30_fixed.json` | 290172 | `e9ea38665bbd8203339b26f9f56b08e8f6e7487d716100565855214ff7bce9fa` | `1e3d7b93772c0521e9bee6dbc37afce5fec354bc0f3dc428b0c960ba4b7227f8` |
| `runs/confirmatory/native-retention-v2/seed30_permuted.json` | 290177 | `a34329e072626460a5baeb52f231e9a4e54a55fb402a2b9b0ca6bcfa771f15c1` | `7c48f2ede1a35f4d9d72d676b38f6051f635701ce4825a227530bb4e8f94e026` |
| `runs/confirmatory/native-retention-v2/seed30_variable.json` | 290195 | `5003609aa89cddc32e24c3cb604e91a58e79c0ee2d064d6fd25a46b84888a801` | `55cc42b6326e21c7fdbe5130d2d8e98b0a2b6048183bed888cb58dfdea43dabd` |
| `runs/confirmatory/native-retention-v2/seed31_fixed.json` | 290172 | `6cd4ddd124f82cf33dfeccb10c6a6f49d36208a5b11ac38f3f0e7a34773026da` | `ca64970b00f3509efc3a5e62b84785cd6f664b53b221b1cc74ed0ece36616a2b` |
| `runs/confirmatory/native-retention-v2/seed31_permuted.json` | 290176 | `2dc8ce68f1741be4871c91120bc1ddce5301e5a09e5455f3cbc6f1d4a6f12bc5` | `2c76710767a8e6e264220b10d1ca9247a881492d2f2070b5b61cf3286acbd987` |
| `runs/confirmatory/native-retention-v2/seed31_variable.json` | 290170 | `0f032037ecf9ee4c43008e775f90527e09d1de1bdf54487683def094a1d1c850` | `4826db33f507b269de0dec6d0a96625d71d3fa19170ba8218521972503bf197a` |
| `runs/confirmatory/native-retention-v2/seed32_fixed.json` | 290190 | `202a91d691622e5d91866f53601624ea705425e1e5345e0ab8bdc57f3af730ef` | `ce159765303344406e98c1236f756d6b3c52b3a8de870d60c715169eaf0ff76d` |
| `runs/confirmatory/native-retention-v2/seed32_permuted.json` | 290180 | `8a226962dfc6c81a2cc26462a0ba2957f7b9a1a50ef2b8c394b8d74c6f6e9a0c` | `5d99a3c490453b6936902f40835109712eca18600311e39579de946f335647eb` |
| `runs/confirmatory/native-retention-v2/seed32_variable.json` | 290175 | `8b2fa9ae4dff0492878837cff9c9519254e8b3178181d97f27d4f192fbf0884a` | `e20441ecd4e9cf23fa395a8706a6abcee7558e406b627b570a098bb3aa039c45` |
| `runs/confirmatory/native-retention-v2/seed33_fixed.json` | 290196 | `959dd258e9869e0579c6e443d44042a7174dee2709cb3e66c1b7353b09d6156e` | `8b7609074c9ac8c4e36e6269eb8b23d45ddfaa91c5c9f09602ff1abbf6e451c1` |
| `runs/confirmatory/native-retention-v2/seed33_permuted.json` | 290197 | `8588dc76d4d56b275594436983ce2a6426f86b049ee36cbc2ceb375c2d0106b2` | `652e0ae6ab9a833f35cb67954616b4c45a1b8a0ad5cd58b7766d3a229324e182` |
| `runs/confirmatory/native-retention-v2/seed33_variable.json` | 290167 | `5e7c36d16a76fdd879a29d8c086d29a621bc6a098d151ccdbf9e3aafa4080030` | `263d1d83656bdd6501b5ff14a3223ce7d8f197f2d8e8dc161586379863052db2` |
| `runs/confirmatory/native-retention-v2/seed34_fixed.json` | 290190 | `f13ff32631f380c58fc1fe769da91944a8f947465a1c0011e7458d6d345f1eba` | `cdc81563f23a3853200525ccf9f320a8684f7e1eca032c5c599b94602a78f0f0` |
| `runs/confirmatory/native-retention-v2/seed34_permuted.json` | 290185 | `fb5dac22c7509b94ddf17415710b5d80ad59414178f29ee5ab9566952a4499a9` | `dfba4904506a36174684bd04c1f4b3ed6f0b95ffa52f982a67d43c3c6fb9ce0d` |
| `runs/confirmatory/native-retention-v2/seed34_variable.json` | 290193 | `a101b03c621676fda86e7c8608ab9ad0179125317938aa4acc95aba4464379bf` | `a7f4e65ca0c29a55bd233545121d46e744a48324e3a271303a531372b6122191` |
| `runs/confirmatory/native-retention-v2/seed35_fixed.json` | 290178 | `53db2e8dcdcd480665ef5b9778ba073e333d1091bc02e2cb7bdba4f85e4f7e2f` | `4ad55b092a99fe56011efce94ac5640a0c5d680e4de778717218f61adbbb30a1` |
| `runs/confirmatory/native-retention-v2/seed35_permuted.json` | 290186 | `f735da4c6dcd72f4840fa5c58fecd3a4cf90c6c941893688c31b0514496f84ca` | `b6f4e600d24f3f9d21b6a792ea1dcc0daf5abc4aba236e487ae59dee09f185d2` |
| `runs/confirmatory/native-retention-v2/seed35_variable.json` | 290183 | `991b240b3650b65e2e2ffb0eb3e6cb77357c89728ce91781338c99aac2d04c3a` | `04bad27c2285b849396b9511d4b702837844fd7ffbca6252a37b050ae06a8bec` |
| `runs/confirmatory/native-retention-v2/seed36_fixed.json` | 290177 | `5303c8f7ae2b767c1ab237008f694399622928780b0a01b820d3e0cb5a8265a2` | `e251672a6da5c4d5c0652c12250b42d15428cc108f5012d83ad1ac521c69da5e` |
| `runs/confirmatory/native-retention-v2/seed36_permuted.json` | 290171 | `3438e0ad33cd2ff56ef603d07d74f457eb913afeac8bc0b42fe2682279dc4a78` | `879f8dd177d63fc60971a150cfa3a4df134fb221bb863f76552916c8d7e977c8` |
| `runs/confirmatory/native-retention-v2/seed36_variable.json` | 290193 | `9f38615bb35237ee1fdb5ef2743c301aea52a2bb0a7a340f43271220db70f50c` | `ff4fd7773823530eeb407cf0b0744c579be9576a9e4b412525be00035cac49b3` |
| `runs/confirmatory/native-retention-v2/seed37_fixed.json` | 290155 | `4e0483fa747d2e79e358cd2261a87f9f915e190a7ce066ac5c1740ca5920602d` | `50d9c444b700b26780f2a6604317bd6fca1fbbaef0cffe9d23700b549359b004` |
| `runs/confirmatory/native-retention-v2/seed37_permuted.json` | 290185 | `feeca8b9e51279c113730f1603bfdd4a01363160bf256ef44dbd7e4aef5616e5` | `7f1d4201eb9eb7210ef907d4428543782cb7e7be4dce05400696667688d3692e` |
| `runs/confirmatory/native-retention-v2/seed37_variable.json` | 290206 | `8dcf76c4037810a7cfb28b82206399dd1c2ee182325a4aceb1742d175c0b7b74` | `eebb952794d4958b36cb3b64849065ccbcf5efdaeefcc6e3f21664a0b102f9ed` |
| `runs/pilot-dev/native-retention-v2/seed0_fixed.json` | 290165 | `e17ae8007aad28d92603001feeb3c457048ac26a1f1a81767110f8f7b1f64dbf` | `542d08a8793e1a96b8190756030a659f911c65b694ed6535a7f8ee2fcd1fe06d` |
| `runs/pilot-dev/native-retention-v2/seed0_permuted.json` | 290156 | `54dc3d9c3a2e7bf02addfadb989013cfffc73b1da3a31534304647bf79697fd1` | `25443fda7402651ed3b7db8c1525a1cb34af70e244d08e4ebfc8f30405f82a77` |
| `runs/pilot-dev/native-retention-v2/seed0_variable.json` | 290195 | `0da524b4a58beda805b2a13deec7641f69ee465e33b74e038bc41a373e375ffd` | `d70835efc622c492eb943e9136582522dfc74d60a0f008d86af72c1a6b900d50` |
| `runs/pilot-dev/native-retention-v2/seed1_fixed.json` | 290183 | `2258afb89db1af92a4b72d44e6c0fd1ac1b66f22083eb98141b1c5283f5590db` | `ba66a7d963f3c708d293587dcb9be82fbdc128e3507446444daf6768f9282f33` |
| `runs/pilot-dev/native-retention-v2/seed1_permuted.json` | 290146 | `5e93a1bad9b8192c0546627103d38318a5786202f29401ad2d9d7ddb7f31f88b` | `8fd4af58d081817cbb33b50455bd64757197105e0112cb79c4b640c73191f2c5` |
| `runs/pilot-dev/native-retention-v2/seed1_variable.json` | 290182 | `c286553f14e2c2c277d1076973c1199870b7a1c2b86fd972741699052d4436de` | `f7e6b662ed6ff48307fd806f06eb9ebbadf26231838367cbf3f7a9b308dcbad6` |
| `runs/pilot-dev/native-retention-v2/seed2_fixed.json` | 290197 | `fba623b290db2af276424e5dcc153cdad182827b1a9401a6f57daead8744cea7` | `c7241d56807b1a307aa5daa51f44449a7643c67eeca897f5e2f761204c456377` |
| `runs/pilot-dev/native-retention-v2/seed2_permuted.json` | 290172 | `55bf56c25122138ae65fe1a231546b39498a7fe6658edad458bedb29b558dcaf` | `05e39f48de5a254954eb19c7b27537888ee4ec546925152ad2bcbed873598f21` |
| `runs/pilot-dev/native-retention-v2/seed2_variable.json` | 290191 | `14948e176ae595af2604ab52acdabd910a201f9c44427af859671a03f6e390ee` | `ebe6b3b100832c86e1cdaa30b47161d083c4470b58adf114b62fc36637553336` |
| `runs/pilot-dev/seed0_fixed.json` | 286958 | `58e5963ebc23ff8f0940523835a441dc0818b20a91039c085febe82b0b8d4d2a` | `38ea86967b6ed6ae29a0a6f1179e4dbfd0569f83328824ff79b900aebdf24411` |
| `runs/pilot-dev/seed0_permuted.json` | 286957 | `623f9d9c12e6dda1df29e3a368877690d3bde0eae372e57499380d8a0518efb4` | `6929affef94c4f098975306385c42a76210ad77258c85e03214d32ec647dedc3` |
| `runs/pilot-dev/seed0_variable.json` | 286974 | `2ab7a0d3530a66a1b4fa99503def49817880775341bba3aa3d541dee7d72ba03` | `f116aa548b1550c58d1e0f55d79bca89da43c9872da7d95ff92ddea0acfc2792` |
| `runs/pilot-dev/seed1_fixed.json` | 286965 | `6a1258ffa1980f42b7a8d6addb0ad0450c8994d294570d84939ef8ade2989c51` | `783dacb761dc453cc514af8fb223d92880626e05177a40c3e2d43bc577341b3c` |
| `runs/pilot-dev/seed1_permuted.json` | 286942 | `300ae49be6ef2de42681ef676ee963b1dc98be1dc5632d0c479380124809a427` | `b10b4b899dc1c6aa981a1c5cc926654bc929edc11c5a8377ad3f275d36bc1de1` |
| `runs/pilot-dev/seed1_variable.json` | 286954 | `fd6b67683196e62d00e12c5cd78dee8f538b21e5dbe894ca70ec750317e08d03` | `8240fc74b11cc8fa946eadf3a52b61c4b651f9901cb804a245984f07232b03f5` |
| `runs/pilot-dev/seed2_fixed.json` | 286972 | `0bc2ebfd3906551ba75a2d844cf4a9d1cb278235f0695cbffa60871f886133d7` | `1c2489ae79700443f3429e52287a54dc12fb8e299478793622cb0adb03df37c2` |
| `runs/pilot-dev/seed2_permuted.json` | 286962 | `38696b81add78400b85385e360501dea8d3e6af465386e79378ef054eca7faeb` | `e18ab734c2fef17038e7b90e94f1d60a527e52b2d86ad7a45c743dc618647540` |
| `runs/pilot-dev/seed2_variable.json` | 286980 | `d1b5f1c03d874fb4fe67c1e51ac8bdbac576b6b583d22b2d72fc81fdd1f08d6e` | `01b36d7ff64e1b820efe7063dbd2e5e5117b27ed887434f1281ed5ae8133e079` |

90 files, 21150037 bytes; 90 of 90 changed by sanitization.
