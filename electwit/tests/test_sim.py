import logging

from electwit.sim import ElecTwit

logger = logging.getLogger(__name__)


def test_simulation():
    """
    Test the election simulation by running a basic simulation and checking the output.
    """

    days = 10

    sim = ElecTwit(
        people_models="deepseek/deepseek-chat:free;7,mistralai/ministral-3b;4,meta-llama/llama-3.1-8b-instruct;4,gemini-2.5-flash-preview-04-17;4",
        candidate_models="deepseek/deepseek-chat:free;1,gemini-2.5-flash-preview-04-17;1",
        eventer_model="deepseek/deepseek-chat:free",
    )

    sim.run_simulation(days=days)

    # Check if the simulation ran without errors

    assert sim.day == days, "Simulation did not run for the expected number of days."
    assert (
        len(sim.polling_history) == days + 1
    ), f"Polling history should contain {days + 1} days: initial and each subsequent day."
