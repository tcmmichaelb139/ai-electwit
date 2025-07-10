import logging

from electwit.sim import ElecTwit

logger = logging.getLogger(__name__)


def test_simulation():
    """
    Test the election simulation by running a basic simulation and checking the output.
    """

    days = 7

    sim = ElecTwit(
        people_models="deepseek/deepseek-chat:free;8",
        candidate_models="deepseek/deepseek-chat:free;2",
    )

    sim.run_simulation(days=days)

    # Check if the simulation ran without errors

    assert sim.day == days, "Simulation did not run for the expected number of days."
    assert (
        len(sim.polling_history) == days + 1
    ), f"Polling history should contain {days + 1} days: initial and each subsequent day."
