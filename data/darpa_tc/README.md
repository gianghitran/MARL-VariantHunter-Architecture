# DARPA TC E3 — pretrain & runtime data

The whole detector pipeline (preprocess → pretrain → main closed-loop eval/anchor/drift)
uses **DARPA TC E3** instead of UNICORN. Raw data (~GB) is git-ignored; download locally.

## Layout
```
data/darpa_tc/<dataset>/
  ta1-<dataset>-e3-official*.json   # provenance (extracted from .tar.gz)
  <dataset>.txt                     # groundtruth: malicious node UUIDs
  <dataset>_edges.pkl               # auto-generated parse cache
```
`<dataset>` ∈ {cadets, theia, trace}. Start with **cadets** (smallest/cleanest).

## 1. Download
```bash
# Groundtruth (automatic) + auto-extract any .tar.gz already present:
python data/darpa_tc/download_darpa.py cadets

# Provenance JSON is on Google Drive (needs file IDs). Either:
#  a) put file IDs in DRIVE_IDS inside download_darpa.py, then re-run the line above; or
#  b) pass an ID directly:
python data/darpa_tc/download_darpa.py cadets --id <gdrive_id> --name ta1-cadets-e3-official-2.json.tar.gz
#  c) download the .tar.gz manually into data/darpa_tc/cadets/ and re-run (auto-extracts).
```
Drive folder (get per-file IDs here):
`https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK`

## 2. Parse + sanity check
```bash
python Benign_Agent/darpa_tc_sampler.py cadets    # parse → cache .pkl, print sample subgraph stats
```

## 3. Pretrain
```bash
python Benign_Agent/pretrain_on_generated.py      # PRETRAIN_SOURCE=darpa (default)
```
Pass: VERIFY shows gap (avg_mal − avg_ben) > ~0.2, F1 > 0.85, low FP.

## 4. Run closed-loop
```bash
python main.py
```
Eval/EWC-anchor/drift reference now come from DARPA automatically.

## Config
- `PRETRAIN_SOURCE` ∈ {darpa, unicorn, attack_agent} (default darpa)
- `DARPA_DATASET` ∈ {cadets, theia, trace} (default cadets) — used by pretrain & main
