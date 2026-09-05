# Sign Language Translator — quarter-dataset build

Rebuilt from your notebook as runnable scripts, using only a quarter of
the WLASL sign classes, plus a FastAPI backend and a simple upload
frontend for inference.

## Pipeline

```
data_prep.py          -> wlasl_quarter.csv        (downloads WLASL, keeps 25% of classes)
extract_all_features.py -> extracted_features/*.npy (MediaPipe keypoints per video)
train.py               -> models/best_sign_language_model.pt, models/vocab.pkl
backend/main.py         -> FastAPI server for inference
frontend/index.html     -> upload a video, see the predicted gloss
```

`model.py`, `vocab.py`, and `extract_features.py` are shared by both the
training scripts and the backend, so the exact same architecture and
feature extraction is used at train time and inference time.

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll need a Kaggle account/API token set up for `kagglehub` to download
the WLASL dataset (same as your original notebook).

## 2. Build the quarter-subset

By default this keeps the **25% of glosses (classes) with the most video
instances** — this gives you more training examples per class than a
random 25% would.

```bash
python data_prep.py --fraction 0.25 --mode top
# or, for a random quarter of the classes instead:
python data_prep.py --fraction 0.25 --mode random
```

This produces `wlasl_quarter.csv`.

## 3. Extract MediaPipe features

```bash
python extract_all_features.py --csv wlasl_quarter.csv --out extracted_features
```

This can take a while depending on how many videos ended up in your
quarter subset — it's safe to re-run (already-extracted `.npy` files are
skipped).

## 4. Train

```bash
python train.py --csv wlasl_quarter.csv --features_dir extracted_features --epochs 50
```

Saves `models/best_sign_language_model.pt` and `models/vocab.pkl`
(needed by the backend).

Useful flags: `--batch_size`, `--hid_dim`, `--epochs`, `--lr` — same
hyperparameters as your notebook, just exposed as CLI args.

## 5. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it's alive: `curl http://localhost:8000/health`

## 6. Open the frontend

Just open `frontend/index.html` directly in a browser (double-click it,
or `open frontend/index.html`). It talks to `http://localhost:8000` by
default — editable at the bottom of the page if your backend runs
elsewhere.

Drop in a short video of a sign, click **Translate sign**, and the
predicted gloss appears.

## Notes / things worth knowing

- **Quarter-subset size**: WLASL has ~2,000 classes total; keeping 25%
  keeps ~500 classes. If that's still more videos than you want to
  process, lower `--fraction` (e.g. `0.1`).
- **CPU-only training** will be slow with the full quarter-subset. Start
  with a smaller `--fraction` (like `0.05`) to confirm the whole pipeline
  works end to end before committing to a bigger training run.
- **CORS** is wide open (`allow_origins=["*"]`) in the backend for local
  development — tighten this before deploying anywhere public.
- The model predicts a **single gloss** per video, matching how WLASL is
  labeled (one sign word per clip), not full sentences.
