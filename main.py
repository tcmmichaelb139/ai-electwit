import logging
import asyncio

from electwit.sim import ElecTwit
from electwit.log_setup import setup_logging

logger = logging.getLogger(__name__)


async def main():
    """
    Main function to run the election simulation.
    """
    setup_logging()

    days = 8

    sim = ElecTwit(
        voter_models="openai/o3-mini;2,google/gemini-2.5-flash;2,anthropic/claude-3.5-haiku;2,deepseek/deepseek-chat-v3-0324;2,qwen/qwq-32b;2,x-ai/grok-3-mini;2,meta-llama/llama-4-maverick;2",
        candidate_models="openai/o3-mini;1,google/gemini-2.5-flash;1",
        eventer_model="google/gemini-2.5-flash",
    )

    await sim.run_simulation(days=days)

    logger.info("Simulation completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
