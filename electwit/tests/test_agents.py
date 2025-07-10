import logging

from electwit.agents import ElectionAgent
from electwit.clients import OpenRouterClient
from electwit.utils import load_prompt, create_random_background

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_agent_diary_consolidation():
    """
    Test the agent's diary consolidation functionality.
    """
    agent = ElectionAgent(
        name="TestAgent",
        client=OpenRouterClient(
            model_name="openrouter/cypher-alpha:free",
            prompt_dir="electwit/prompts",
        ),
        role="voter",
        background=create_random_background(),
    )

    agent.client.set_system_prompt(
        load_prompt("system_prompt_voter.txt", "electwit/prompts/voter")
    )

    diary = [
        "I support lowering taxes.",
        "I don't like Bob because he supports higher government spending.",
        "Joe does bring up a good point about reducing government waste.",
        "Government spending should be reduced.",
        "I'm not sure about Alice's stance on healthcare.",
    ]

    for entry in diary:
        agent.add_today_diary_entry(entry)

    agent.consolidate_diary()

    logger.debug(f"Consolidated Diary: {agent.consolidated_diary}")
    assert len(agent.consolidated_diary) > 0, "Consolidated diary should not be empty."
