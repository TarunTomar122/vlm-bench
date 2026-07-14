from vlm_bench.heldout_builder import color_subject, choice_answer, modal_answer, stratified_select


def test_choice_answer_returns_semantic_choice() -> None:
    assert choice_answer("(B)", ["left", "right"]) == "right"


def test_modal_answer_is_deterministic_on_ties() -> None:
    assert modal_answer(["Blue", "red", "blue", "red"]) == ("blue", 2)


def test_color_subject_extracts_presence_target() -> None:
    assert color_subject("Is the traffic light red in this image?") == "traffic light"


def test_stratified_select_is_image_disjoint_and_excludes_reference() -> None:
    rows = [
        {"image_sha256": "excluded", "stable_key": "0", "locator": "p:0", "group": "a"},
        {"image_sha256": "one", "stable_key": "1", "locator": "p:1", "group": "a"},
        {"image_sha256": "one", "stable_key": "2", "locator": "p:2", "group": "b"},
        {"image_sha256": "two", "stable_key": "3", "locator": "p:3", "group": "b"},
    ]
    selected = stratified_select(rows, 2, lambda row: row["group"], {"excluded"})
    assert {row["image_sha256"] for row in selected} == {"one", "two"}
