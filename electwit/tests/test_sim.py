import logging

from electwit.sim import ElecTwit

logger = logging.getLogger(__name__)


def test_simulation():
    """
    Test the election simulation by running a basic simulation and checking the output.
    """

    days = 10

    sim = ElecTwit(
        people_models="openai/gpt-4o-mini;4,mistralai/devstral-medium;4,meta-llama/llama-4-maverick;4,x-ai/grok-3-mini;4",
        candidate_models="openai/gpt-4o-mini;1,x-ai/grok-3-mini;1",
        eventer_model="gemini-2.5-flash-lite-preview-06-17",
    )

    sim.run_simulation(days=days)

    # Check if the simulation ran without errors

    assert sim.day == days, "Simulation did not run for the expected number of days."
    assert (
        len(sim.polling_history) == days + 1
    ), f"Polling history should contain {days + 1} days: initial and each subsequent day."
