import itertools
from pathlib import Path

import numpy as np
import pytest

import expression_content_pilot as pilot
import run_task_form_battery as parent


def test_complete_crossed_protocol_and_unchanged_parent_statements():
    source=parent.load_json(parent.DEFAULT_PROTOCOL)
    protocol=pilot.build_protocol(source)
    assert len(protocol["stimuli"])==160
    originals={x["stimulus_id"]:x for x in parent.expand_protocol(source)}
    for row in protocol["stimuli"]:
        original=originals[row["parent_stimulus_id"]]
        assert row["statement"]==original["statement"]
        assert row["truth"]==original["truth_value"]
        assert row["prompt"].endswith("\nOdpověď:")
    cells={tuple(r[a] for a in pilot.AXES) for r in protocol["stimuli"]}
    assert cells==set(itertools.product(*(protocol["axes"][a] for a in pilot.AXES)))


def test_five_factor_decomposition_separates_output_truth_interaction():
    shape=(3,2,2,2,2,4)
    country=np.arange(3).reshape(3,1,1,1,1,1)*np.array([1,0,0,0])
    truth=np.array([-1,1]).reshape(1,1,2,1,1,1)
    code=np.array([-1,1]).reshape(1,1,1,1,2,1)
    x=np.broadcast_to(country + 2*truth*code*np.array([0,1,0,0]),shape)
    result=pilot.decompose(x)
    assert result["ss"]["truth"]==pytest.approx(0)
    assert result["ss"]["output_code"]==pytest.approx(0)
    assert result["ss"]["truthxoutput_code"]>0
    assert sum(result["share"].values())==pytest.approx(1)
    assert all(v==pytest.approx(0) for k,v in result["ss"].items() if k not in ("country","truthxoutput_code"))


def test_truth_transfer_detects_shared_direction_and_opposite_code_mapping():
    x=np.zeros((4,2,2,2,2,3))
    for t in range(2):x[:,:,t,:,:,0]=1 if t==0 else -1
    assert all(r["cosine"]==pytest.approx(1) for r in pilot.truth_transfer(x))
    x[:,:,:,:,1]*=-1
    assert all(r["cosine"]==pytest.approx(-1) for r in pilot.truth_transfer(x))


def test_readout_keeps_truth_code_and_unconstrained_token_decisions_distinct():
    item={"stimulus_id":"toy","country":"x","relation":"capital","truth":"true","input_form":"v1","output_code":"cs"}
    ids={"cs_true":0,"cs_false":1,"en_true":2,"en_false":3}
    # Correct truth within the prescribed code, but the other code wins globally.
    rows=pilot.readout_rows(item,ids,np.array([[3.,1.,5.,0.,7.]]))
    assert rows[0]["within_code_truth_correct"]
    assert not rows[0]["four_label_code_correct"]
    assert not rows[0]["full_vocab_top1_correct"]
    assert 0<rows[0]["selected_code_probability"]<1


def test_incomplete_factorial_cells_are_rejected():
    source=parent.load_json(parent.DEFAULT_PROTOCOL);protocol=pilot.build_protocol(source)
    items=protocol["stimuli"][:-1]
    with pytest.raises(ValueError,match="factorial cells"):
        pilot.ordered_tensor(protocol,items,{x["stimulus_id"]:np.ones((3,4)) for x in items})
