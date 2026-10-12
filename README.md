# LN 2026 Project: "Help the doctor"

Goal: classify each clinical text into one of **12 medical specialties**, then write a **3-page paper** using `Project-Template/template.tex`.

**Deadline: Friday 16 October 2026, 23:59, via Fenix.**

---

## 0. How you are graded

| Part | Points | What you need |
|---|---|---|
| Automatic: accuracy | 2.0 | Accuracy on `test_no_labels.csv` **> 49%** (that beats the MNB + TF-IDF(description) baseline) |
| Automatic: per-class F1 | 2.0 | In addition to the above, **every class F1 ≥ 25%** (no class can be ignored) |
| Paper | 16.0 | Breakdown in step 11 |
| Penalty | −3.0 | Applied if **any** instruction is not followed (file names, page limit, zip format, and so on) |

You get **zero on the automatic part** if you cannot reproduce your own `results.txt`. Keep the code deterministic (fix the seeds) and runnable.

---

## 1. Final deliverables checklist

Submit `NUM.zip` (**zip, not rar**). NUM is your group number. It contains:

- [ ] `NUM.pdf`: the paper (3 pages max; the bibliography and a 1-page appendix do not count)
- [ ] the code (**no trained model files**)
- [ ] any extra datasets you collected or generated (for example, augmented data)
- [ ] `results.txt`: one predicted label per line, **no header**, in the same order as `test_no_labels.csv`

---

## 2. Suggested repo layout

```
Projeto-LN-2026/
├── train.csv
├── test_no_labels.csv
├── src/
│   ├── data.py          # loading + cleaning + split (step 1, 3)
│   ├── eda.py           # stats + plots for the paper (step 2)
│   ├── baseline.py      # MNB baseline reproduction (step 4)
│   ├── augment.py       # data augmentation (step 7)
│   ├── models/
│   │   ├── tfidf_linear.py   # M1
│   │   ├── <second>.py       # M2
│   │   └── transformer.py    # M3
│   ├── evaluate.py      # metrics, confusion matrix, error dump (step 8, 9)
│   └── predict.py       # final model -> results.txt (step 10)
├── outputs/             # figures, tables, misclassified CSVs (not submitted models)
├── paper/               # LaTeX (copy of Project-Template)
├── results.txt
└── README.md
```

---

## 3. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn nltk matplotlib seaborn
pip install torch transformers datasets accelerate   # only for the transformer model
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

---

## 4. Step by step

### Step 1: Load and clean the data (pay attention to the details below)

Things already found in the files:

| Issue | Fix |
|---|---|
| Separator is `;`. Fields contain line breaks inside quotes, so `wc -l` gives the wrong count | `pd.read_csv(path, sep=';')` |
| Labels have long runs of trailing spaces | `df.medical_specialty = df.medical_specialty.str.strip()` |
| 3 rows in `train.csv` are malformed and their "label" is a piece of transcription | Drop rows whose label is not one of the 12 classes |
| **`test_no_labels.csv` has NO header row** (409 rows, 4 columns) | `pd.read_csv(path, sep=';', header=None, names=['description','sample_name','transcription','keywords'])` |
| Missing values. Train: keywords ≈405, transcription 33, sample_name 7, description 3. Test: keywords 83, transcription 11 | `.fillna('')` |

The 12 classes and their training counts: Surgery 876, Cardiovascular-Pulmonary 299, Orthopedic 292, Radiology 217, General Medicine 207, Gastroenterology 192, Neurology 181, Obstetrics-Gynecology 135, Neurosurgery 78, Ophthalmology 71, Psychiatry-Psychology 40, Dermatology 25.

Use the same preprocessing function for train and test.

### Step 2: Exploratory data analysis (the paper needs at least 2 concrete observations)

Produce figures and tables, and **read actual examples**. Starting observations already found:

1. **Strong imbalance**: Surgery has 876 rows and Dermatology has 25, a ratio of about 35×. Surgery is 33% of the data, so always predicting "Surgery" already gives about 33% accuracy.
2. **Duplicated notes with conflicting labels**: 799 transcriptions are duplicates. Of the 702 duplicate groups, **701 contain different labels for the same text** (for example, a knee operation labelled both *Surgery* and *Orthopedic*). This comes from the original Kaggle MTSamples data. It sets a ceiling on achievable accuracy and is the biggest source of errors. Discuss it.
3. **Keyword leakage/cue**: in about 70% of rows the first keyword equals the specialty name (`"neurology, temporal lobe, ..."`). Keywords are missing in ~15% of rows, so the model cannot rely on them alone.
4. Length: description median ~16 words; transcription median ~387 words, max ~2450. That matters for transformers, which accept at most 512 tokens.
5. Also look at: overlapping classes (Neurology vs Neurosurgery, Surgery vs everything else, General Medicine vs Cardio). Pick a few samples per class and read them.

Save the plots (class distribution bar chart, length histogram) to `outputs/`.

### Step 3: Make your own evaluation split

- `test_no_labels.csv` is **only** for generating `results.txt`. Never use it for training, tuning or analysis.
- Split `train.csv` into **train / dev / test** (e.g. 80/10/10), stratified by label, with a fixed seed (`random_state=42`).
- **Group the duplicates**: put all copies of the same transcription in the same split (use `StratifiedGroupKFold` with `groups = transcription`). Otherwise the dev/test scores are inflated by leakage. Mention this in the paper.
- Optionally run 5-fold CV for more stable numbers (the minority classes are tiny).

### Step 4: Reproduce the baseline

Multinomial Naive Bayes + TF-IDF on **description only**, with lowercasing and stemming (NLTK `PorterStemmer`). Report its accuracy and macro-F1 on your split. This is the reference line in your results table.

### Step 5: Choose the research question (it drives the whole paper)

Pick **one** explicit question before choosing models. Examples:

- *"How do classical sparse models compare to fine-tuned clinical transformers on small, imbalanced, noisily-labelled clinical text?"*
- *"Which data augmentation strategy best improves minority-class F1 in imbalanced clinical text classification?"*
- *"Which input fields (description, sample_name, transcription, keywords) carry the most signal for specialty classification?"*

The models, experiments and discussion should each help answer it.

### Step 6: Build at least 3 models

Suggested set (adjust it to fit the research question):

| # | Model | Notes |
|---|---|---|
| M1 | TF-IDF (word 1–2-grams + char 3–5-grams) over **all fields concatenated** → `LinearSVC` / `LogisticRegression` with `class_weight='balanced'` | Fast and strong. Tune `C` and `min_df`. Likely the best accuracy/effort trade-off |
| M2 | Per-field TF-IDF (`ColumnTransformer`, separate vectorizer per field, field weights) or a soft-voting ensemble | Lets you run a field ablation (is description alone enough? keywords?) |
| M3 | Fine-tuned clinical transformer: `emilyalsentzer/Bio_ClinicalBERT` or `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | Input: `description [SEP] sample_name [SEP] keywords [SEP] transcription`, truncated to 512 tokens (try head+tail). Use a weighted loss for imbalance. Training on CPU/MPS is slow, so keep epochs at 3–5 |
| M4 (optional) | Zero/few-shot LLM, or sentence embeddings + LogReg | Adds to the creativity score |

Stop words: **do not remove them blindly** (the PDF warns about it). If you remove them, compare with and without.

For replicability, log every hyperparameter, seed, library version and preprocessing step.

### Step 7: Data augmentation (mandatory, 2 points)

Augment **only the training split**, never dev/test. Ideas, mostly aimed at the minority classes (Dermatology, Psychiatry, Ophthalmology, Neurosurgery):

- **EDA-style operations**: synonym replacement (WordNet), random swap/deletion (Wei & Zou, 2019).
- **Back-translation** EN→PT→EN with MarianMT (`Helsinki-NLP/opus-mt-en-ROMANCE`, `opus-mt-ROMANCE-en`).
- **Section/sentence shuffling or cropping** of transcriptions (clinical notes have sections like `HISTORY:`, `PLAN:`). Random sub-spans of the note make new samples.
- **LLM paraphrasing** of minority-class descriptions. If you do this, declare it in the paper and save the generated data in the zip.

Run an experiment **with vs. without** augmentation (and ideally compare 2 strategies). Report the effect on macro-F1 and on minority-class F1.

### Step 8: Evaluate

For every model, on your own test split:

- accuracy, **macro-F1** (more informative than accuracy under imbalance), weighted-F1
- per-class precision / recall / F1 for the best model (`classification_report`)
- a confusion matrix with **readable labels** (short names, rotated ticks) for at least one model
- check that **every class has F1 ≥ 0.25**. If one does not, try class weights, more augmentation or threshold tweaks

### Step 9: Error analysis (Discussion is worth 3 points)

- Dump misclassified rows to `outputs/errors_<model>.csv` (text excerpt, gold, predicted, confidence).
- **Read them.** Look for patterns, for example:
  - Surgery vs. a surgical specialty (Orthopedic, Neurosurgery, ObGyn): is the label really wrong, or ambiguous?
  - duplicates with conflicting labels
  - Radiology reports for organs of another specialty
- Pick **≥3 concrete examples**. For each, write the excerpt, predicted label, gold label and your interpretation.

### Step 10: Final model → `results.txt`

1. Retrain the best configuration on **all** of `train.csv` (plus augmentation), using the same hyperparameters.
2. Predict `test_no_labels.csv` (read it with `header=None`).
3. Write one label per line, with no header and no index:
   ```python
   with open('results.txt', 'w') as f:
       f.write('\n'.join(preds) + '\n')
   ```
4. Sanity checks:
   ```python
   lines = open('results.txt').read().splitlines()
   assert len(lines) == 409               # == len(test df)
   assert set(lines) <= set(CLASSES)      # labels stripped, spelled exactly like train
   ```
   Also check that the prediction distribution looks plausible (not 90% Surgery).

### Step 11: Write the paper (16 points)

**Use LaTeX.** The PDF says the paper must follow the provided template, and that template is `Project-Template/template.tex`. The easiest route is **Overleaf**: upload `template.tex` and `biblio.bib`, then compile. `pdflatex` is not installed on this Mac; installing locally is optional (`brew install --cask mactex-no-gui`).

Rules:
- **3 pages max**. Only the first 3 pages are read, so do **not** add a cover page. The bibliography and **one appendix page** (extra figures/tables only) are excluded from the limit. The paper must make sense without the appendix.
- Author line: group number, names, and the **% contribution of each member (mandatory)**.
- English or Portuguese, formal (no "it's"), no subjective adjectives, label and reference every figure/table, define acronyms once, use LaTeX quotes ``like this''.
- **Delete the "Evaluation and tips" section** before submitting.

What each section must contain:

| Section | Pts | Content |
|---|---|---|
| Introduction | 1.5 | Task + **explicit research question** |
| Data Analysis | 1.5 | Class distribution, stats, **≥2 concrete observations** (step 2) |
| Data Augmentation | 2.0 | Clear description of the strategy (step 7) |
| Models | 4.0 | ≥3 models, choices justified by the research question, **reproducible** detail, creativity |
| Experimental Setup | 1.0 | Datasets, your splits, metrics, hyperparameters (explain each briefly) |
| Results | 1.5 | Best models table, per-class results of best model, confusion matrix of one model |
| Discussion | 3.0 | Common errors, **≥3 misclassified examples** (excerpt / pred / gold / interpretation), link back to the research question, label-noise discussion |
| Future Work | 0.5 | Next steps tied to the research question |
| References | 1.0 | All papers, libraries, pre-trained models, **plus an explicit statement of how LLMs were used** |

Suggested references: MTSamples Kaggle dataset; scikit-learn (Pedregosa et al., 2011); NLTK; BERT (Devlin et al., 2019); ClinicalBERT (Alsentzer et al., 2019); EDA (Wei & Zou, 2019); MarianMT (Junczys-Dowmunt et al., 2018) if used.

### Step 12: Package

```bash
mkdir NUM && cp -r src README.md results.txt NUM/ && cp paper/NUM.pdf NUM/
# + any augmented / extra datasets
zip -r NUM.zip NUM
```
Then check that the zip opens, contains no model weights, and that `results.txt` has 409 lines.

---

## 5. Suggested 4-day schedule

| Day | Work |
|---|---|
| Mon 12 | Steps 1–4: load, EDA, split, baseline. Pick the research question |
| Tue 13 | Steps 6–7: M1, M2, augmentation experiments. Start M3 training |
| Wed 14 | Finish M3. Steps 8–10: evaluation, error analysis, `results.txt`. Start writing |
| Thu 15 | Write the paper (Overleaf), figures, check page limit |
| Fri 16 | Proofread, final checklist, zip, **submit before 23:59** |

Everyone in the group should understand the whole pipeline, because an **oral discussion** may be held.

Questions go to `meic-ln@disciplinas.tecnico.ulisboa.pt` (subject: *Project*). Check the Fenix FAQs.
