import logging
import asyncio

from electwit.sim import ElecTwit

from electwit.log_setup import setup_logging

logger = logging.getLogger(__name__)


def test_simulation():
    """
    Test the election simulation by running a basic simulation and checking the output.
    """

    setup_logging()

    days = 10

    # test
    # sim = ElecTwit(
    #     limits=15,
    #     memory_limit=5,
    #     people_models="deepseek/deepseek-chat-v3-0324:free;2",
    #     candidate_models="deepseek/deepseek-chat-v3-0324:free;2",
    #     eventer_model="deepseek/deepseek-chat-v3-0324:free",
    #     candidate_similarity=[-1.0, -0.75],
    # )
    # people_models="meta-llama/llama-3.2-3b-instruct;2",
    # candidate_models="meta-llama/llama-3.2-3b-instruct;2",
    # eventer_model="meta-llama/llama-3.2-3b-instruct",

    sim = ElecTwit(
        voter_models="openai/gpt-4.1-mini;2,google/gemini-2.5-flash;2,anthropic/claude-3.5-haiku;2,deepseek/deepseek-chat-v3-0324;2,qwen/qwq-32b;2,x-ai/grok-3-mini;2,moonshotai/kimi-k2;2,mistralai/devstral-medium;2",
        candidate_models="google/gemini-2.5-flash;1,openai/gpt-4.1-mini;1",
        eventer_model="google/gemini-2.5-flash",
        candidate_similarity=[-1.0, -0.75],
        limits=15,
        memory_limit=5,
    )

    asyncio.run(sim.run_simulation(days=days))

    # Check if the simulation ran without errors

    assert sim.day == days, "Simulation did not run for the expected number of days."
