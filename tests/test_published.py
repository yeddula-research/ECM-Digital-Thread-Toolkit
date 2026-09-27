import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecm_digital_thread import config as C
from ecm_digital_thread import published as P


def test_table_shapes():
    assert P.TABLE1.shape == (5, 5)
    assert P.TABLE2.shape == (6, 4)
    assert P.TABLE3.shape == (4, 6)
    assert P.TABLE4.shape == (4, 5)


def test_table1_improvements_follow_from_before_after():
    check = P.table1_improvement_check()
    assert check["Difference (pp)"].abs().max() <= 0.1


def test_table2_f1_is_harmonic_mean_of_precision_and_recall():
    check = P.table2_f1_check()
    assert check["Difference"].abs().max() <= 0.005


def test_table2_overall_average_is_mean_of_classes():
    for col, v in P.table2_average_check().items():
        assert abs(v["mean_of_classes"] - v["reported"]) <= 0.005, col


def test_table4_proposed_row_matches_table1():
    assert all(v["match"] for v in P.table4_matches_table1().values())


def test_demonstration_scenario_is_independent_of_the_paper():
    reported = P.TABLE1.set_index("Metric")
    for metric, points in C.DEMO_SCENARIO.items():
        row = reported.loc[metric]
        assert points["month_1"] != float(row["Before Implementation"]), metric
        assert points["month_6"] != float(row["After Implementation"]), metric
    assert C.CCT_BASELINE_DAYS != float(reported.loc["Change Cycle Time (days)", "Before Implementation"])


def test_disclosure_sentence_is_defined():
    assert "synthetic" in C.DATASET_DISCLOSURE
    assert "not the paper's case-study data" in C.DATASET_DISCLOSURE
