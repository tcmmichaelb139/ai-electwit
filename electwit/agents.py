import os
import logging

from typing import Optional, List

from electwit.utils import load_prompt, apply_background_prompt
from electwit.clients import BaseModelClient

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent for election agent and eventer agent"""

    def __init__(
        self,
        name: str,
        client: BaseModelClient,
        role: str,
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
        self.chance_to_act: float = chance_to_act
        self.prompt_dir: Optional[str] = prompt_dir

        self.journal: List[str] = []  # used for debugging
        self.today_diary: List[str] = []  # short term memory (of the day)
        self.consolidated_diary: List[str] = []  # long term memory

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
            logger.info(f"{self.name} has no diary entries to consolidate.")
            return ""

        consolidated_input = "\n\n".join(self.consolidated_diary[-10:])
        today_diary_input = "\n\n".join(self.today_diary)

        consolidation_prompt = load_prompt("consolidation_prompt.txt", self.prompt_dir)

        consolidated_response = self.client.generate_response(
            consolidation_prompt.format(
                name=self.name,
                role=self.role,
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

    def formatted_consolidated_diary(self, limit: int = 10) -> str:
        """Returns the consolidated diary entries formatted for the prompt."""
        if not self.consolidated_diary:
            logger.info(
                f"{self.name} has no consolidated diary entries when formatting for prompt."
            )
            return "(No consolidated diary entries)"

        consolidated_diary = "\n\n".join(self.consolidated_diary[-limit:])
        if not consolidated_diary:
            logger.info(f"{self.name} has no consolidated diary entries to return.")
            return "(No consolidated diary entries)"

        return consolidated_diary

    def formatted_today_diary(self) -> str:
        """Returns the today's diary entries formatted for the prompt."""
        if not self.today_diary:
            logger.info(
                f"{self.name} has no today's diary entries when formatting for prompt."
            )
            return "(No today's diary entries)"

        today_diary = "\n\n".join(self.today_diary)
        if not today_diary:
            logger.info(f"{self.name} has no today's diary entries to return.")
            return "(No today's diary entries)"

        return today_diary


class ElectionAgent(BaseAgent):
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
        super().__init__(name, client, role, chance_to_act, prompt_dir)

        self.background: str = background

        self.actions_in_hour: List[dict] = (
            []
        )  # actions taken by the agent in the current hour

        logger.info(
            f"Initialized {self.role} ElectionAgent: {self.name} with model {self.client.model_name}"
        )
        self.add_journal_entry(
            f"Initialized {self.role} ElectionAgent: {self.name} with model {self.client.model_name}"
        )

    def act_posting(
        self,
        hour: int,
        day: int,
        candidates: str,
        polling_numbers: str,
        current_feed: str,
        recent_events: str,
    ):
        """Simulates the posting/discussion phase"""

        logger.info(
            f"{self.name} ({self.role}) is acting (Posting and Discussion) at day {day}, hour {hour}."
        )

        content_prompt = load_prompt("content_prompt.txt", self.prompt_dir).format(
            name=self.name,
            current_phase="Posting and Discussion",
            current_day=day,
            current_hour=hour,
            background=apply_background_prompt(self.background),
            candidates=candidates,
            consolidated_diary=self.formatted_consolidated_diary(limit=7),
            todays_diary=self.formatted_today_diary(),
            prior_poll_numbers=polling_numbers,
            current_feed=current_feed,
            recent_events=recent_events,
        )

        actions_prompt = load_prompt("action_prompt_posting.txt", self.prompt_dir)

        prompt = content_prompt + "\n\n" + actions_prompt

        response = self.client.generate_response_json_list(prompt=prompt)

        self.actions_in_hour = response[:10] if response else []
        self._validate_actions()

        self.post_act_posting(
            day=day,
            hour=hour,
            content_prompt=content_prompt,
            actions=self.actions_in_hour,
        )

    def _validate_actions(self):
        """validates the actions and removes any invalid actions"""

        self.actions_in_hour = [
            action for action in self.actions_in_hour if self._valid_action(action)
        ]

    def _valid_action(self, action: dict) -> bool:
        """validates a single action"""

        if not isinstance(action, dict):
            logger.warning(f"Invalid action format: {action}. Expected a dictionary.")
            return False

        if (
            "action" in action
            and "content" in action
            and "id" not in action
            and action["action"] == "POST"
        ):
            return True
        elif (
            "action" in action
            and "content" in action
            and "id" in action
            and action["action"] == "REPLY"
        ):
            return True
        elif "action" in action and "id" in action and action["action"] == "LIKE":
            return True

        logger.warning(
            f"Invalid action: {action}. Must be a valid action with 'action', 'content', and optionally 'id'."
        )
        return False

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
            "action_prompt_posting_diary_entry.txt", self.prompt_dir
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

    def act_polling(
        self,
        hour: int,
        day: int,
        candidates: str,
        polling_numbers: str,
        current_feed: str,
        recent_events: str,
    ) -> dict[str, str]:
        """Simulates the polling phase"""

        logger.info(f"{self.name} ({self.role}) is acting (Polling) at day {day}.")

        content_prompt = load_prompt("content_prompt.txt", self.prompt_dir).format(
            name=self.name,
            current_phase="Posting and Discussion",
            current_day=day,
            current_hour=hour,
            background=apply_background_prompt(self.background),
            candidates=candidates,
            consolidated_diary=self.formatted_consolidated_diary(limit=7),
            todays_diary=self.formatted_today_diary(),
            prior_poll_numbers=polling_numbers,
            current_feed=current_feed,
            recent_events=recent_events,
        )

        actions_prompt = load_prompt("action_prompt_polling.txt", self.prompt_dir)

        prompt = content_prompt + "\n\n" + actions_prompt

        response = self.client.generate_response_json_list(prompt=prompt)[0]

        return response

    def get_todays_actions(self) -> List[dict]:
        return self.actions_in_hour

    def name_and_background(self) -> str:
        """returns the name and background"""
        return (
            f"Name: {self.name}\nBackground: {apply_background_prompt(self.background)}"
        )

    def __str__(self):
        return f"""----------
Name: {self.name}
Model: {self.client.model_name}
Role: {self.role}
Chance to act: {self.chance_to_act}
Background: {apply_background_prompt(self.background)}
---------"""


class Event:
    """
    Represents an event
    """

    def __init__(self, content: str, day: int, hour: int):
        """inits event"""

        self.content = content
        self.day = day
        self.hour = hour

    def __str__(self):
        return f"Event on day {self.day}, hour {self.hour}: {self.content}"


class EventorAgent(BaseAgent):
    """
    Represents an event creator (eventer)
    Also stores the events
    """

    def __init__(
        self,
        name: str,
        client: BaseModelClient,
        role: str,
        chance_to_act: float = 0.5,
        prompt_dir: Optional[str] = None,
    ):
        super().__init__(name, client, role, chance_to_act, prompt_dir)

        self.events: List[Event] = []

        self.specific_events: dict[int, str] = {}
        self._set_up_specific_events()

    def _set_up_specific_events(self):
        """Sets up specific events for the eventer agent"""
        specific_events_folder = f"{self.prompt_dir}/eventer/specific_events"
        specific_events_files = [
            f for f in os.listdir(specific_events_folder) if f.endswith(".txt")
        ]

        for file in specific_events_files:
            if file.endswith("_day.txt"):
                specific_event = load_prompt(file, specific_events_folder)

                day = int(file.split("_")[0])

                if day in self.specific_events:
                    logger.warning(
                        f"Duplicate specific event for day {day} in {file}. Overwriting existing event."
                    )

                self.specific_events[day] = specific_event

    def _remove_specific_event(self, day: int):
        """Removes a specific event for the given day"""
        if day in self.specific_events:
            del self.specific_events[day]
            logger.info(f"Removed specific event for day {day}.")
        else:
            logger.warning(f"No specific event found for day {day} to remove.")

    def create_event(
        self, day: int, hour: int, recent_poll: str, feed: str, event_limit: int
    ) -> Event:
        """Creates event"""

        logger.info(f"Eventer creating event: Day: {day}, Hour: {hour}")

        content_prompt = load_prompt(
            "content_prompt_eventer.txt", self.prompt_dir + "/eventer"
        ).format(
            day=day,
            hour=hour,
            poll=recent_poll,
            previous_events=self.get_formatted_events(limit=event_limit),
            today_diary=self.formatted_today_diary(),
            consolidated_diary=self.formatted_consolidated_diary(),
        )

        logger.debug(f"Content prompt for event creation: {content_prompt}")

        specific_events = self.specific_events.get(day, "(No specific event)")

        action_prompt = load_prompt(
            "action_prompt_eventer.txt", self.prompt_dir + "/eventer"
        ).format(specific_events=specific_events)

        self._remove_specific_event(day)  # remove specific event after using it

        response = self.client.generate_response(
            content_prompt + "\n\n" + action_prompt
        )

        if not response:
            logger.error(
                f"Failed to generate event content for {self.name} ({self.role}). Response was empty."
            )
            return Event(content="(No event occurred)", day=day, hour=hour)

        event = Event(content=response, day=day, hour=hour)
        self.events.append(event)

        logger.info(f"Event created: {event.content} on Day {day}, Hour {hour}")

        # add diary entry after creating the event

        diary_entry_prompt = load_prompt(
            "action_prompt_eventer_diary_entry.txt", self.prompt_dir + "/eventer"
        ).format(event=str(event))

        diary_response = self.client.generate_response(
            content_prompt + "\n\n" + diary_entry_prompt
        )

        if not diary_response:
            logger.error(
                f"Failed to generate diary entry for event for {self.name} (Day {day}, Hour {hour}). Response was empty."
            )
            diary_response = "(No diary entry generated)"

        self.add_today_diary_entry(f"Day: {day}, Hour: {hour}: {diary_response}")

        return event

    def get_formatted_events(self, limit: int) -> str:
        """Formats the events for the prompt"""
        if not self.events:
            logger.info(f"{self.name} has no events to format.")
            return "(No events)"

        formatted_events = "\n\n".join(str(event) for event in self.events[-limit:])

        return formatted_events

    def __str__(self):
        return f"""----------
Name: {self.name}
Model: {self.client.model_name}
Role: {self.role}
Chance to act: {self.chance_to_act}
---------"""
