import json

import pytest
from PIL import Image

from georc.conditions import CONDITIONS
from georc.data import Pair, Round
from georc.generate import (
    build_messages,
    chain_path,
    pending_rounds,
    pilot_rounds,
    restrict_pairs,
    write_chain,
)
from georc.judge import load_template


def make_rounds(challenges: list[str], per_challenge: int = 5) -> list[Round]:
    return [
        Round(challenge=c, index=i, country="kg", image=f"{c}_{i}.png")
        for c in challenges
        for i in range(1, per_challenge + 1)
    ]


def test_baseline_conditions_use_the_paper_prompt_on_each_model():
    assert (CONDITIONS["G0"].model, CONDITIONS["G0"].prompt) == ("gemma3-12b", "gen_paper.txt")
    assert (CONDITIONS["Q0"].model, CONDITIONS["Q0"].prompt) == ("qwen2.5-vl-7b", "gen_paper.txt")
    assert CONDITIONS["G0"].mm_kwargs is None


def test_chain_path_layout(tmp_path):
    rnd = make_rounds(["abc"])[1]
    assert chain_path(tmp_path, "G0", 2, rnd) == tmp_path / "chains" / "G0" / "seed2" / "abc_2.txt"


def test_pilot_is_the_first_twenty_challenges_in_name_order():
    names = [f"c{n:03d}" for n in range(30, 0, -1)]  # unsorted on purpose
    rounds = make_rounds(sorted(names))
    pilot = pilot_rounds(rounds)
    assert len(pilot) == 100
    assert {r.challenge for r in pilot} == {f"c{n:03d}" for n in range(1, 21)}


def test_restrict_pairs_keeps_only_pairs_for_the_given_rounds():
    rounds = make_rounds(["a", "b"], per_challenge=1)
    pairs = [Pair(round=r, expert="Human_Expert_1", chain="x") for r in rounds]
    assert [p.round.challenge for p in restrict_pairs(pairs, rounds[:1])] == ["a"]


def test_write_chain_writes_text_and_metadata_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "chains" / "G0" / "seed0" / "abc_1.txt"
    write_chain(path, "Cyrillic signage\nConclusion: Kyrgyzstan", {"output_tokens": 12})
    assert path.read_text() == "Cyrillic signage\nConclusion: Kyrgyzstan"
    assert json.loads(path.with_suffix(".json").read_text()) == {"output_tokens": 12}
    assert sorted(p.name for p in path.parent.iterdir()) == ["abc_1.json", "abc_1.txt"]


def test_pending_rounds_skips_finished_chains_and_ignores_partial_temp_files(tmp_path):
    rounds = make_rounds(["abc"], per_challenge=3)
    write_chain(chain_path(tmp_path, "G0", 0, rounds[0]), "done", {})
    stray = chain_path(tmp_path, "G0", 0, rounds[1]).with_suffix(".txt.tmp")
    stray.write_text("half written by a preempted job")
    assert [r.index for r in pending_rounds(rounds, tmp_path, "G0", 0)] == [2, 3]


def test_build_messages_sends_the_image_then_the_prompt_text():
    image = Image.new("RGB", (4, 4))
    [message] = build_messages("gen_paper.txt", image)
    assert message["role"] == "user"
    image_part, text_part = message["content"]
    assert image_part == {"type": "image_pil", "image_pil": image}
    assert text_part == {"type": "text", "text": load_template("gen_paper.txt")}


def test_unknown_condition_is_rejected():
    with pytest.raises(KeyError):
        CONDITIONS["Z9"]
