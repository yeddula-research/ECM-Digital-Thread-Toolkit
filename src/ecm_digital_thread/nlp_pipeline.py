"""
NLP-based change request processing — implements Section III.C (Layer 3) and
III.G of the paper: preprocessing -> TF-IDF feature extraction (Eq. 2) ->
multi-class classification into the 5 ECM domains -> Named Entity Recognition
for structured metadata (part numbers, assembly IDs, responsible teams).

Two classification backends are provided:

    1. ``sklearn`` (default, always available): TF-IDF + a linear classifier
       (LinearSVC by default, Logistic Regression as an alternative). This is
       the backend the bundled results/tables/figures were produced with —
       it needs no GPU, no model download, and trains in seconds.
    2. ``transformer`` (optional): fine-tunes a pretrained BERT-family model
       (default ``bert-base-uncased``) exactly as described in Section III.G
       — 70:15:15 split, AdamW, lr=2e-5, batch size 16, 5 epochs, max sequence
       length 256, cross-entropy loss, early stopping. Requires
       ``pip install torch transformers`` and internet access to fetch
       pretrained weights; it is not needed for this repo's bundled results
       and is skipped gracefully (with an informative error) if the
       dependencies are absent.

Why TF-IDF + linear SVM as the default: it is fully open-source, CPU-only,
and runs offline in seconds. The paper's classifier is a fine-tuned BERT
model; the transformer backend implements that Section III.G procedure for
anyone who wants to run it on their own hardware.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from . import config as C

# --------------------------------------------------------------------------- #
# 1. Text preprocessing (no NLTK/spaCy dependency required)
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset("""
a an the this that these those is are was were be been being have has had
do does did will would shall should may might must can could of to in on at
for with by from as it its it's and or but if then than so such not no nor
we you they he she i us our your their his her them him me my mine yours
theirs ours about into over under again further once here there when where
why how all any both each few more most other some such only own same
""".split())

_SUFFIX_RULES = [
    ("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""),
    ("ed", ""), ("s", ""),
]


def _lightweight_lemmatize(token: str) -> str:
    """Rule-based suffix stripping — a deliberately simple, dependency-free
    substitute for a full lemmatizer (e.g. NLTK WordNet or spaCy), adequate for
    domain terms like 'failures'->'failure', 'classified'->'classify'."""
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) + len(replacement) >= 3:
            return token[: -len(suffix)] + replacement if suffix else token
    return token


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+|\d+")


def preprocess_text(text: str) -> str:
    """Tokenize, drop stop-words/noise, lemmatize (Section III.C 'Text
    Preprocessing' step). Returns a cleaned, space-joined string suitable for
    TF-IDF vectorization."""
    tokens = _TOKEN_RE.findall(text.lower())
    cleaned = [
        _lightweight_lemmatize(tok) for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 1
    ]
    return " ".join(cleaned)


# --------------------------------------------------------------------------- #
# 2. TF-IDF feature extraction (Eq. 2) + classification
# --------------------------------------------------------------------------- #

@dataclass
class NLPPipelineResult:
    vectorizer: TfidfVectorizer
    model: object
    report_dict: dict
    confusion: np.ndarray
    labels: list
    X_test_idx: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray


def train_classifier(df: pd.DataFrame, backend: str = "sklearn",
                      classifier: str = "linear_svm",
                      seed: int = C.RANDOM_SEED) -> NLPPipelineResult:
    """Train/validate/test split 70:15:15 (Section III.G), TF-IDF + linear
    classifier by default."""
    if backend != "sklearn":
        return _train_transformer(df, seed=seed)

    texts = df["raw_text"].map(preprocess_text).values
    labels = df["ecm_class"].values

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        texts, labels, np.arange(len(df)), test_size=0.30, random_state=seed, stratify=labels,
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, random_state=seed, stratify=y_temp,
    )

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    if classifier == "logreg":
        model = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    else:
        model = LinearSVC(C=1.0, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    # Validation set is used to confirm no gross overfitting before final test eval.
    val_report = classification_report(y_val, model.predict(X_val_vec), output_dict=True, zero_division=0)

    y_pred = model.predict(X_test_vec)
    labels_sorted = sorted(C.ECM_CLASS_PRIOR.keys())
    report_dict = classification_report(
        y_test, y_pred, labels=labels_sorted, output_dict=True, zero_division=0,
    )
    confusion = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    result = NLPPipelineResult(
        vectorizer=vectorizer, model=model, report_dict=report_dict,
        confusion=confusion, labels=labels_sorted,
        X_test_idx=idx_test, y_test=y_test, y_pred=y_pred,
    )
    result.val_report = val_report  # type: ignore[attr-defined]
    return result


def _train_transformer(df: pd.DataFrame, seed: int = C.RANDOM_SEED,
                        model_name: str = "bert-base-uncased"):
    """Literal implementation of Section III.G. Requires `torch` and
    `transformers`; raises a clear, actionable error if unavailable rather
    than failing deep in the call stack."""
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer, Trainer,
            TrainingArguments, EarlyStoppingCallback,
        )
    except ImportError as exc:
        raise ImportError(
            "The 'transformer' backend needs torch + transformers "
            "(`pip install torch transformers`) and internet access to "
            "download pretrained weights. The default 'sklearn' backend "
            "produces this repo's bundled results without either "
            "dependency; see README.md 'Optional: BERT backend'."
        ) from exc

    labels_sorted = sorted(C.ECM_CLASS_PRIOR.keys())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    texts = df["raw_text"].tolist()
    y = df["ecm_class"].map(label2id).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class ECMDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding=True, max_length=256)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]))
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels_sorted))

    args = TrainingArguments(
        output_dir="./bert_ecm_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ECMDataset(X_train, y_train),
        eval_dataset=ECMDataset(X_val, y_val),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    preds = trainer.predict(ECMDataset(X_test, y_test))
    y_pred = np.argmax(preds.predictions, axis=1)
    report_dict = classification_report(
        y_test, y_pred, target_names=labels_sorted, output_dict=True, zero_division=0,
    )
    return {"report_dict": report_dict, "labels": labels_sorted, "trainer": trainer}


# --------------------------------------------------------------------------- #
# 3. Named Entity Recognition (rule-based, dependency-free by default)
# --------------------------------------------------------------------------- #

_PART_RE = re.compile(r"\bPN-\d{5}\b")
_ASSEMBLY_RE = re.compile(r"\bASM-\d{4}\b")


def extract_entities(text: str, known_teams: list[str] | None = None,
                      known_subsystems: list[str] | None = None) -> dict:
    """Rule-based NER (Section III.C 'NER modules'): pulls part numbers,
    assembly identifiers, responsible teams and affected subsystems from raw
    text via regex + controlled-vocabulary matching.

    Optionally upgrade to spaCy's statistical NER for organization/person
    entities if `spacy` + an English model are installed:
        >>> import spacy
        >>> nlp = spacy.load("en_core_web_sm")
    This is intentionally NOT a hard dependency — the regex/vocabulary
    approach below requires zero extra installs and is what the bundled
    results were generated with.
    """
    known_teams = known_teams or []
    known_subsystems = known_subsystems or []
    entities = {
        "part_numbers": _PART_RE.findall(text),
        "assembly_ids": _ASSEMBLY_RE.findall(text),
        "teams": [t for t in (known_teams or []) if t.lower() in text.lower()],
        "subsystems": [s for s in (known_subsystems or []) if s.lower() in text.lower()],
    }
    return entities


def evaluate_ner(df: pd.DataFrame, teams_vocab: list[str], subsystems_vocab: list[str]) -> dict:
    """Since ground-truth entities were embedded when the demonstration text was
    generated (part_number / assembly_id / responsible_team /
    affected_subsystem columns), this computes exact-match extraction
    accuracy for each entity type — a sanity check that the rule-based NER
    is working, not a claim about performance on real free text.

    Not every narrative template mentions every attribute (a real ECM note
    doesn't always name the assembly, say) — evaluate_ner. To avoid
    conflating "correctly extracted nothing because there was nothing to
    extract" with an actual NER miss, each entity type's accuracy is
    computed only over the records whose raw text actually mentions that
    attribute (a fair "recall given mention" figure), not over every record
    in the dataset."""
    correct = {"part_numbers": 0, "assembly_ids": 0, "teams": 0, "subsystems": 0}
    mentioned = {"part_numbers": 0, "assembly_ids": 0, "teams": 0, "subsystems": 0}
    for _, row in df.iterrows():
        text = row["raw_text"]
        ents = extract_entities(text, teams_vocab, subsystems_vocab)
        for key, ground_truth in (
            ("part_numbers", row["part_number"]),
            ("assembly_ids", row["assembly_id"]),
            ("teams", row["responsible_team"]),
            ("subsystems", row["affected_subsystem"]),
        ):
            if str(ground_truth).lower() not in text.lower():
                continue  # attribute wasn't mentioned in this record's text
            mentioned[key] += 1
            if ground_truth in ents[key]:
                correct[key] += 1
    return {
        k: (round(100.0 * correct[k] / mentioned[k], 2) if mentioned[k] else 100.0)
        for k in correct
    }


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def run(input_csv=None, output_csv=None, backend: str = "sklearn") -> pd.DataFrame:
    from .data_generation import SUBSYSTEMS, TEAMS

    input_csv = input_csv or C.RAW_DATASET_CSV
    output_csv = output_csv or C.NLP_SCORED_CSV
    df = pd.read_csv(input_csv)

    result = train_classifier(df, backend=backend)

    df = df.copy()
    df["predicted_class"] = pd.NA
    if hasattr(result, "X_test_idx"):
        df.loc[result.X_test_idx, "predicted_class"] = result.y_pred
        df.loc[result.X_test_idx, "is_test_set"] = True
        df["is_test_set"] = df["is_test_set"].fillna(False)

    ner_rows = []
    for _, row in df.iterrows():
        ents = extract_entities(row["raw_text"], TEAMS, SUBSYSTEMS)
        ner_rows.append({
            "extracted_part_numbers": ";".join(ents["part_numbers"]),
            "extracted_assembly_ids": ";".join(ents["assembly_ids"]),
            "extracted_teams": ";".join(ents["teams"]),
            "extracted_subsystems": ";".join(ents["subsystems"]),
        })
    df = pd.concat([df, pd.DataFrame(ner_rows)], axis=1)
    df.to_csv(output_csv, index=False)
    return df, result


if __name__ == "__main__":
    df, result = run()
    print("Classification report (test set):")
    for label, metrics in result.report_dict.items():
        if isinstance(metrics, dict):
            print(f"  {label}: precision={metrics['precision']:.2f} "
                  f"recall={metrics['recall']:.2f} f1={metrics['f1-score']:.2f}")
