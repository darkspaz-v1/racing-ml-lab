import numpy as np
import pytest

from racing_lab.decision_lab import DecisionProbe
from racing_lab.network import Network


def test_probe_changes_only_its_frozen_input_and_outputs():
    network = Network(np.random.default_rng(7))
    network.w1[:] = 0
    network.w2[:] = 0
    network.w1[3, 0] = 2
    network.w2[0, 0] = 1
    original_weights = network.w1.copy()
    observation = np.zeros(12, dtype=np.float32)
    probe = DecisionProbe.capture(network, observation, actual_action=2)
    probe.set_input(3)
    probe.set_value(0.75)
    _, baseline, _ = probe.baseline()
    _, edited, predicted = probe.evaluate()
    assert edited[0] > baseline[0]
    assert predicted == 0
    assert probe.actual_action == 2
    assert observation[3] == 0
    assert np.array_equal(network.w1, original_weights)
    probe.set_value(10)
    assert probe.edited[3] == 1
    probe.set_input(8)
    probe.set_value(-10)
    assert probe.edited[8] == -1
    probe.reset()
    assert np.array_equal(probe.edited, observation)


def test_probe_rejects_incomplete_observations():
    network = Network(np.random.default_rng(7))
    with pytest.raises(ValueError):
        DecisionProbe.capture(network, [0, 1])
