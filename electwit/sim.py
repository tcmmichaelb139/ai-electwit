import os
import logging
from time import sleep
import random

from typing import List, Optional

from electwit.platform import Platform
from electwit.agents import ElectionAgent
from electwit.clients import load_model_client
from electwit.utils import (
    load_prompt,
    random_new_name,
    create_random_background,
    random_number,
    get_closest_response,
)

SEED = os.getenv("SEED")

logger = logging.getLogger(__name__)

random.seed(SEED)


class ElecTwit:
    def __init__(
        self,
        people_models: str = "openrouter/cypher-alpha:free;10",
        candidate_models: str = "openrouter/cypher-alpha:free;2",
    ):
        """
        Initialize the ElecTwit simulation.

        num_people (int): Number of people in the simulation.
        people_models (str): Model names for the people, formatted as "model_name;count,model_name;count,".
        num_candidates (int): Number of candidates in the election.
        candidate_models (str): Model names for the candidates, formatted as "model_name;count,model_name;count,".
        """

        self.people_models = people_models.split(",")
        self.candidate_models = candidate_models.split(",")

        self.people_agents = self.initialize_agents(self.people_models, "voter")
        self.candidate_agents = self.initialize_agents(
            self.candidate_models, "candidate", different=True
        )

        self._log_people()
        self._log_candidates()

        self.day = 0
        self.polling_history = []
        self.platform = Platform()

    def initialize_agents(
        self, models: List[str], role: str, different: Optional[bool] = False
    ) -> list[ElectionAgent]:
        """
        Initialize agents for the given models.
        """

        agents = []
        for model in models:
            model_name, count = model.split(";")

            if not model_name or not count.isdigit():
                raise ValueError(
                    f"Invalid model format: {model}. Expected 'model_name;count'."
                )

            count = int(count)
            for _ in range(count):
                client = load_model_client(model_name, "electwit/prompts")

                client.set_system_prompt(
                    load_prompt(f"system_prompt_{role}.txt", f"electwit/prompts/{role}")
                )

                background = create_random_background()

                if different:
                    pass

                agents.append(
                    ElectionAgent(
                        name=random_new_name(),
                        client=client,
                        role=role,
                        background=background,
                        chance_to_act=random_number(0.5, 1),
                        prompt_dir=os.path.join("electwit", "prompts"),
                    )
                )

        return agents

    def run_polling(self):
        """
        Runs an initial polling to get current preferences
        """

        results = {"ABSTAIN": 0}

        for candidate in self.candidate_agents:
            results[candidate.name] = 0

        for agent in self.people_agents + self.candidate_agents:
            vote = agent.act_polling(
                hour=0, day=0, candidates=self._get_candidates_names()
            )

            action = get_closest_response(
                testing_text=vote["action"],
                responses=["VOTE", "ABSTAIN"],
            )

            if action == "VOTE":
                candidate = get_closest_response(
                    testing_text=vote["candidate"],
                    responses=self._get_candidates_names_list(),
                )
                results[candidate] += 1

                agent.add_journal_entry(f"Voted for {vote['candidate']}")
            elif action == "ABSTAIN":
                results["ABSTAIN"] += 1

                agent.add_journal_entry(f"Abstained from voting")

        logger.info(f"Polling results Day {self.day}: {results}")

        self.polling_history.append(results)

    def run_day(self):
        """
        Simulates a day in the election simulation.
        Each agent has a % chance to interact with the platform.
        """

        self.day += 1

        for hour in range(9, 18):  # Simulating from 9 AM to 5 PM
            combined = self.people_agents + self.candidate_agents
            random.shuffle(combined)  # Shuffle to randomize order of actions
            for agent in combined:
                if random_number(0, 1) < agent.chance_to_act:
                    agent.act_posting(
                        hour=hour,
                        day=self.day,
                        candidates=self._get_candidates_names(),
                        polling_numbers=self._get_most_recent_polling(),
                        current_feed=self.platform.get_feed_as_string(limit=10),
                    )

                    actions = agent.get_todays_actions()

                    self.platform.apply_actions(agent.name, self.day, hour, actions)

    def end_day(self):
        """
        Ends the day in the simulation.

        What it does:
        - Consolidates the diaries of all agents.
        - Runs polling
        """

        self.run_polling()

        for agent in self.people_agents + self.candidate_agents:
            agent.consolidate_diary(self.day)

    def final_polling(self) -> dict[str, int]:
        """
        Returns the final polling results as a dictionary.
        """

        if not self.polling_history:
            return {"No polling history available.": 0}

        latest_poll = self.polling_history[-1]
        return latest_poll

    def _get_candidates_names_list(self) -> List[str]:
        return [candidate.name for candidate in self.candidate_agents]

    def _get_candidates_names(self) -> str:
        """
        Returns a list of candidate names.
        """
        return ", ".join(self._get_candidates_names_list())

    def _get_most_recent_polling(self) -> str:
        """
        Returns the most recent polling results as a string.
        """

        if not self.polling_history:
            return "No polling history available."

        latest_poll = self.polling_history[-1]
        return "; ".join(
            [f"{candidate}: {votes} votes" for candidate, votes in latest_poll.items()]
        )

    def run_simulation(self, days: int = 7):
        """Runs the entire simulation for a specified number of days."""

        try:
            self.run_polling()
            for _ in range(days):
                logger.info(f"Starting day {self.day + 1} of the simulation.")
                self.run_day()
                self.end_day()

            logger.info("Final Polling Results:")
            for candidate in self.candidate_agents:
                votes = sum(
                    poll.get(candidate.name, 0) for poll in self.polling_history
                )
            logger.info(f"{candidate.name}: {votes} votes")
        except Exception as e:
            logger.error(f"An error occurred during the simulation: {e}")
            raise
        finally:
            logger.info("Simulation completed. Saving results...")

            # log the platform state and agent states and polling

            with open("simulation_results.txt", "w") as f:
                f.write("--- Platform State: ---\n")
                f.write(str(self.platform))
                f.write("\n\n---Polling History:---\n")
                for day, poll in enumerate(self.polling_history, start=1):
                    f.write(f"Day {day}: {poll}\n")

                f.write("\n\n--- People Agents: ---\n")
                for agent in self.people_agents:
                    f.write(
                        str(agent) + "\n"
                        f"Consolidated Diary: {agent.formatted_consolidated_diary()}\n"
                        f"Today's Diary: {agent.formatted_today_diary()}\n"
                    )

                f.write("\n\n--- Candidate Agents: ---\n")
                for agent in self.candidate_agents:
                    f.write(
                        str(agent) + "\n"
                        f"Consolidated Diary: {agent.formatted_consolidated_diary()}\n"
                        f"Today's Diary: {agent.formatted_today_diary()}\n"
                    )

    def _log_people(self):
        logger.info("People in the simulation:")
        for agent in self.people_agents:
            logger.info(f"Person: {agent.name}, Political View: {agent.background}")

    def _log_candidates(self):
        logger.info("Candidates in the simulation:")
        for agent in self.candidate_agents:
            logger.info(f"Candidate: {agent.name}, Political View: {agent.background}")
