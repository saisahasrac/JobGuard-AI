# JobGuard AI — Starter Repository

A student project starter for detecting potentially fraudulent job and internship postings.
It uses the EMSCAD dataset (`fake_job_postings.csv`), TF-IDF and Logistic Regression, with a Streamlit demo.

## 1. Project structure

```text
JobGuard_AI/
├── data/
│   └── fake_job_postings.csv        # add the downloaded EMSCAD CSV here
├── models/                          # generated after training
├── outputs/                         # generated evaluation files
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 2. Setup

Use Python 3.10 or 3.11 if possible.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

Put the downloaded dataset at:

`data/fake_job_postings.csv`

The target column must be named `fraudulent` (0 = legitimate, 1 = fraudulent).

## 3. Train the model

From the repository root:

```bash
python -m src.train
```

The script creates:
- `models/jobguard_model.joblib`
- `models/threshold.json`
- `outputs/test_metrics.json`
- `outputs/test_predictions.csv`

It uses a stratified train/validation/test split. The validation set is used to select a decision threshold; the test set is held out for final evaluation.

## 4. Run the web app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## 5. Team workflow (5 people)

1. Dataset & EDA: inspect data, missing values, class balance and plots.
2. NLP preprocessing: own `src/preprocessing.py`; coordinate changes before editing it.
3. Model & evaluation: own `src/train.py`, metrics and saved model.
4. Streamlit UI: own `app.py`; can build the interface while model training is underway.
5. Explainability, risk indicators, integration and report: test the full flow, maintain documentation and prepare report/PPT.

Agree on the column names and function interface before parallel coding. Commit small changes and test before merging.

## Important limitations

- EMSCAD is a historical dataset and is not a live database of current Indian job offers.
- A model score is a model-derived risk indicator, not proof that a posting is fraudulent and not necessarily a calibrated real-world probability.
- Rule-based indicators (for example, requests for upfront payment) are warning signs, not definitive evidence.
- Do not enter real personal information, bank details, passwords or OTPs into the demo.
- Report precision, recall and F1 for the fraudulent class; accuracy alone can hide poor fraud detection because the classes are imbalanced.
