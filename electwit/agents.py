import logging

from typing import Optional, List

from electwit.utils import load_prompt, apply_background_prompt
from electwit.clients import BaseModelClient

logger = logging.getLogger(__name__)


class ElectionAgent:
    """
    Represents an agent (voter/candidate) in the simulation.
    """

    def __init__(
        self,
        name: str,
        client: BaseModelClient,
        role: str,
        background: str,
        chance_to_act: float = 0.5,
        prompt_dir: Optional[str] = None,
    ):
        """
        Initializes ElectionAgent

        Args:
        - name (str): Name of the agent.
        - client (BaseModelClient): The client used for generating responses.
        - role (str): Role of the agent, either 'voter' or 'candidate
        - prompt_dir (Optional[str]): Directory for loading prompts. Defaults to None.
        """
        self.name: str = name
        self.client: BaseModelClient = client
        self.role: str = role
        self.background: str = background
        self.chance_to_act: float = chance_to_act
        self.prompt_dir: Optional[str] = prompt_dir

        self.journal: List[str] = []  # used for debugging
        self.today_diary: List[str] = []  # short term memory (of the day)
        self.consolidated_diary: List[str] = []  # long term memory

        self.actions_in_hour: List[dict] = (
            []
        )  # actions taken by the agent in the current hour

        logger.info(
            f"Initialized {self.role} agent: {self.name} with model {self.client.model_name}"
        )
        self.add_journal_entry(
            f"Initialized {self.role} agent: {self.name} with model {self.client.model_name}"
        )

    def add_journal_entry(self, entry: str):
        """Adds string entry to the agent's journal for debugging purposes."""

        if not entry:
            raise ValueError("Journal entry cannot be empty.")

        self.journal.append(entry)
        logger.debug(f"Journal entry added for {self.name}: {entry}")

    def add_today_diary_entry(self, entry: str):
        """Adds string entry to the agent's diary."""

        if not entry:
            raise ValueError("Diary entry cannot be empty.")

        self.today_diary.append(entry)
        logger.debug(f"Diary entry added for {self.name}: {entry}")

    def clear_today_diary(self):
        """Clears the agent's diary."""
        logger.info(f"Clearing diary for {self.name} ({self.role})")
        self.today_diary.clear()

    def add_consolidated_diary_entry(self, entry: str):
        """Adds string entry to the agent's consolidated diary."""

        if not entry:
            raise ValueError("Consolidated diary entry cannot be empty.")

        self.consolidated_diary.append(entry)
        logger.debug(f"Consolidated diary entry added for {self.name}: {entry}")

    def consolidate_diary(self, day: int):
        """Consolidates the diary entries into a single string."""
        logger.info(f"Consolidating diary entries for {self.name} ({self.role})")
        if not self.today_diary:
            logger.warning(f"{self.name} has no diary entries to consolidate.")
            return ""

        consolidated_input = "\n\n".join(self.consolidated_diary[-10:])
        today_diary_input = "\n\n".join(self.today_diary)

        consolidation_prompt = load_prompt("consolidation_prompt.txt", self.prompt_dir)

        consolidated_response = self.client.generate_response(
            consolidation_prompt.format(
                name=self.name,
                role=self.role,
                background=apply_background_prompt(self.background),
                consolidated_diary=consolidated_input,
                today_diary=today_diary_input,
            )
        )

        if not consolidated_response:
            logger.error(
                f"Failed to consolidate diary entries for {self.name} ({self.role}). Response was empty."
            )
            return ""

        logger.info(
            f"Consolidated diary entries for {self.name} ({self.role}): {consolidated_response}"
        )

        self.clear_today_diary()
        self.add_consolidated_diary_entry(f"Day {day}: {consolidated_response}")

    def formatted_consolidated_diary(self) -> str:
        """Returns the consolidated diary entries formatted for the prompt."""
        if not self.consolidated_diary:
            logger.warning(
                f"{self.name} has no consolidated diary entries when formatting for prompt."
            )
            return "(No consolidated diary entries)"

        consolidated_diary = "\n\n".join(self.consolidated_diary[-10:])
        if not consolidated_diary:
            logger.warning(f"{self.name} has no consolidated diary entries to return.")
            return "(No consolidated diary entries)"

        return consolidated_diary

    def formatted_today_diary(self) -> str:
        """Returns the today's diary entries formatted for the prompt."""
        if not self.today_diary:
            logger.warning(
                f"{self.name} has no today's diary entries when formatting for prompt."
            )
            return "(No today's diary entries)"

        today_diary = "\n\n".join(self.today_diary)
        if not today_diary:
            logger.warning(f"{self.name} has no today's diary entries to return.")
            return "(No today's diary entries)"

        return today_diary

    def act_posting(
        self,
        hour: int,
        day: int,
        candidates: str,
        polling_numbers: str,
        current_feed: str,
    ):
        """Simulates the posting/discussion phase"""

        logger.info(
            f"{self.name} ({self.role}) is acting (Posting and Discussion) at hour {hour}."
        )

        content_prompt = load_prompt(
            "content_prompt_posting.txt", self.prompt_dir
        ).format(
            name=self.name,
            current_phase="Posting and Discussion",
            current_day=day,
            current_hour=hour,
            background=apply_background_prompt(self.background),
            candidates=candidates,
            consolidated_diary=self.formatted_consolidated_diary(),
            todays_diary=self.formatted_today_diary(),
            prior_poll_numbers=polling_numbers,
            current_feed=current_feed,
        )

        actions_prompt = load_prompt("actions_posting.txt", self.prompt_dir)

        response = self.client.generate_response_json_list(
            prompt=content_prompt + "\n\n" + actions_prompt,
        )

        self.actions_in_hour = response if response else []

        self.post_act_posting(
            day=day,
            hour=hour,
            content_prompt=content_prompt,
            actions=self.actions_in_hour,
        )

    def post_act_posting(
        self, day: int, hour: int, content_prompt: str, actions: list[dict]
    ):
        """
        Post action for the posting phase
        - Get diary entry
        """

        actions_prompt = ""
        for action in actions:
            if "id" in action:
                if "content" in action:
                    actions_prompt += f"{action['action']} (ID: {action['id']}): {action['content']}\n"
                else:
                    actions_prompt += f"{action['action']} (ID: {action['id']})\n"
            else:
                actions_prompt += f"{action['action']}: {action['content']}\n"

        action_diary_entry = load_prompt(
            "actions_diary_entry.txt", self.prompt_dir
        ).format(
            actions=actions_prompt,
        )

        response = self.client.generate_response(
            content_prompt + "\n\n" + action_diary_entry
        )

        if not response:
            logger.error(
                f"Failed to generate diary entry for actions in posting phase for {self.name} ({self.role}). Response was empty."
            )
            return {}

        self.add_today_diary_entry(f"Day: {day}, Hour: {hour}: {response}")

    def act_polling(self, hour: int, day: int, candidates: str) -> dict[str, str]:
        """Simulates the polling phase"""

        logger.info(f"{self.name} ({self.role}) is acting (Polling) at hour {hour}.")

        content_prompt = load_prompt(
            "content_prompt_polling.txt", self.prompt_dir
        ).format(
            name=self.name,
            current_phase="Polling",
            current_day=day,
            current_hour=hour,
            background=apply_background_prompt(self.background),
            candidates=candidates,
            consolidated_diary=self.formatted_consolidated_diary(),
            todays_diary=self.formatted_today_diary(),
        )

        actions_prompt = load_prompt("actions_polling.txt", self.prompt_dir)

        response = self.client.generate_response_json_list(
            prompt=content_prompt + "\n\n" + actions_prompt
        )[0]

        return response

    def get_todays_actions(self) -> List[dict]:
        return self.actions_in_hour

    def __str__(self):
        return f"name: {self.name} ({self.client.model_name}), role: {self.role}, chance to act: {self.chance_to_act}"
