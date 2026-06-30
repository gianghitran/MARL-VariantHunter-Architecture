# Experiments — MARL-VariantHunter (RQ1 / RQ2 / RQ3)

Harness danh gia he thong MARL phat hien APT. Tat ca ket qua duoc gop vao MOT file
de so sanh: `results/all_results.csv` + `results/summary.md`.

## Nguyen tac

- **Mot test corpus dong bang, mot ham eval.** Moi checkpoint (B0 pretrained, B1/C*/A*
  co-evolved) deu danh gia qua `common.eval_detector` tren cung corpus -> so sanh fair.
- **Camouflage = evasion that su cua he thong.** B0 phat hien 100% mau "sach" -> de co
  mau B0 BO SOT (FN0, can cho Recovery Rate), corpus tron benign padding quanh APT
  (`datasets.camouflage_corpus`) dung y co che environment.py (benign mimicry / LOTL).
- **Toi thieu so lan chay closed-loop MARL = 4** nho cache + tai dung:

| Run | Config | Phuc vu |
|---|---|---|
| canonical | gen ON, EWC ON, clf=mlp, variants=aligned | RQ1-B1, RQ2-C2, RQ3-A0 |
| rq2c1     | gen OFF, EWC ON, clf=mlp                  | RQ2-C1 |
| rq3linear | gen ON, EWC ON, clf=linear               | RQ3-A1 |
| rq3deep   | gen ON, EWC ON, clf=deep                 | RQ3-A3 |

B0 va C0 KHONG chay MARL (== detector pretrained, chi inference).

## Cach chay

Dung python cua project (vd conda env `nmuggle_APT`). Chay tu thu muc goc repo.

```bash
PY=/home/tmkhiem/.conda/envs/nmuggle_APT/bin/python

# Smoke (nhanh, validate pipeline — 1 seed, episodes nho):
$PY experiments/rq1_marl_vs_nomarl.py --episodes 20
$PY experiments/rq2_unseen_variants.py --episodes 20
$PY experiments/rq3_classifier_swap.py --episodes 20
$PY experiments/consolidate.py
cat experiments/results/summary.md

# Full (bo --episodes -> hard-cap 150 ep cua main.py):
$PY experiments/rq1_marl_vs_nomarl.py
$PY experiments/rq2_unseen_variants.py
$PY experiments/rq3_classifier_swap.py
$PY experiments/consolidate.py
```

Cac script chia se cache MARL theo `tag` (xem `run_marl.py`), nen canonical run chi
chay MOT lan du RQ1/RQ2/RQ3 cung tham chieu. Chay lai voi `force=True` trong code neu
muon train lai.

## Mo rong N seed

Chay lai moi script voi `--seed <s>` khac nhau (tag MARL gom seed nen khong dung cache
nham). `consolidate.py` tu dong tinh mean±std khi co nhieu seed — khong can sua code.

```bash
for s in 42 1 2; do
  $PY experiments/rq1_marl_vs_nomarl.py --seed $s
done
$PY experiments/consolidate.py
```

## Env-var hooks (mac dinh tat -> hanh vi goc khong doi)

| Var | Tac dung | Doc o |
|---|---|---|
| `EXP_SEED` | seed toan cuc (deterministic) | main.py |
| `EXP_NUM_EPISODES` | ghi de hard-cap 150 episodes | main.py |
| `EXP_RUN_DIR` | thu muc output run (thay cho runs/<ts>) | main.py |
| `EXP_DISABLE_GENERATION` | tat Generation Agent (Action 0 = no-op) | main.py / environment.py |
| `EXP_DISABLE_EWC` | bo qua EWC continual learning | main.py |
| `EXP_CLASSIFIER` | head classifier {mlp\|linear\|deep} | environment.py + networks.make_classifier |
| `EXP_ALLOWED_VARIANTS` | gioi han variant Generation sinh (CSV) | Attack_Agent/generation_interface.py |
| `EXP_ALLOWED_APTS` | gioi han APT Generation sinh (CSV) | Attack_Agent/generation_interface.py |

## File

```
common.py      # seed, load_detector, eval_detector, metrics, write_raw_result
datasets.py    # frozen corpus: DARPA + APT variants + camouflage; build_rq{1,2}_corpus
run_marl.py    # cached wrapper quanh main.py; canonical_run/c1_run/classifier_run; parse run.log
rq1_*.py rq2_*.py rq3_*.py   # orchestrators -> results/raw/*.json
consolidate.py # results/raw/*.json -> all_results.csv + summary.md
results/
  raw/         # 1 json / dieu kien
  cache/       # corpus DARPA da sample (tranh parse pkl 3.4GB lai)
  manifests/   # metadata corpus (audit/reproduce)
  marl_runs/<tag>/   # output tung MARL run (best_mlp.pth, run.log)
```

## Luu y (1 seed / smoke)

Voi 1 seed, `summary.md` hien gia tri tho (chua co y nghia thong ke). De dua vao paper
nen chay 3–5 seed (chi can lap lai script voi `--seed`). Corpus DARPA dung seed co dinh
(`datasets.DARPA_SEED`) nen test set giong nhau giua cac seed model.
