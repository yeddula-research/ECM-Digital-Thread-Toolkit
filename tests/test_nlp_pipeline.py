import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecm_digital_thread.data_generation import generate_dataset
from ecm_digital_thread.nlp_pipeline import (
    preprocess_text, extract_entities, train_classifier, evaluate_ner,
)


def test_preprocess_removes_stopwords_and_lowercases():
    cleaned = preprocess_text("The Quick Brown Fox is jumping over the lazy dogs")
    tokens = cleaned.split()
    assert "the" not in tokens
    assert "is" not in tokens
    assert "quick" in tokens
    assert "jump" in tokens or "jumping" in cleaned  # suffix-stripped or retained, but present


def test_extract_entities_finds_part_and_assembly():
    text = "NCR: part PN-12345 on assembly ASM-6789 failed inspection."
    ents = extract_entities(text, known_teams=["Quality Assurance"], known_subsystems=["hydraulic pump"])
    assert "PN-12345" in ents["part_numbers"]
    assert "ASM-6789" in ents["assembly_ids"]


def test_extract_entities_matches_known_vocabulary():
    text = "The hydraulic pump issue was escalated to Quality Assurance."
    ents = extract_entities(text, known_teams=["Quality Assurance", "Field Service"],
                             known_subsystems=["hydraulic pump", "alternator"])
    assert "Quality Assurance" in ents["teams"]
    assert "hydraulic pump" in ents["subsystems"]
    assert "Field Service" not in ents["teams"]


def test_classifier_beats_random_baseline():
    df = generate_dataset(n_records=600, seed=11)
    result = train_classifier(df)
    # 5 classes -> random baseline macro F1 ~ 0.20; classifier should do far better.
    macro_f1 = result.report_dict["macro avg"]["f1-score"]
    assert macro_f1 > 0.5


def test_ner_evaluation_returns_percentages_in_range():
    from ecm_digital_thread.data_generation import SUBSYSTEMS, TEAMS
    df = generate_dataset(n_records=200, seed=12)
    acc = evaluate_ner(df, TEAMS, SUBSYSTEMS)
    for key, val in acc.items():
        assert 0.0 <= val <= 100.0
