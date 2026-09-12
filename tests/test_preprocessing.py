from scripts.build_pathway_hypergraph import map_wsi_fname_to_sample_id
from scripts.make_expression_and_mask import choose_one_sample_per_case
from kp2surv.data import index_case_files_lexicographically


def test_expression_sample_selection_is_deterministic():
    samples = [
        "TCGA-AB-1234-01A-02R",
        "TCGA-AB-1234-01A-01R",
        "TCGA-AB-1234-02A-01R",
    ]

    assert choose_one_sample_per_case(samples) == "TCGA-AB-1234-01A-01R"
    assert choose_one_sample_per_case(list(reversed(samples))) == "TCGA-AB-1234-01A-01R"


def test_wsi_expression_match_uses_lexicographic_tie_break():
    samples = [
        "TCGA-AB-1234-01A-02R",
        "TCGA-AB-1234-01A-01R",
        "TCGA-AB-1234-02A-01R",
    ]
    selected, case_id, match_type = map_wsi_fname_to_sample_id(
        "TCGA-AB-1234-01Z-00-DX1",
        set(samples),
        {"TCGA-AB-1234": list(reversed(samples))},
    )

    assert selected == "TCGA-AB-1234-01A-01R"
    assert case_id == "TCGA-AB-1234"
    assert match_type == "case_prefix_primary"


def test_multiple_wsi_selection_is_lexicographic():
    filenames = [
        "TCGA-AB-1234-02Z-00-DX1.pt",
        "TCGA-AB-1234-01Z-00-DX1.pt",
        "TCGA-CD-5678-01Z-00-DX1.pt",
    ]

    selected = index_case_files_lexicographically(filenames)

    assert selected["TCGA-AB-1234"] == "TCGA-AB-1234-01Z-00-DX1.pt"
