import os
import json
import logging
import time
import random

from typing import List, Optional

from electwit.log_setup import get_log_dir
from electwit.platform import Platform
from electwit.agents import ElectionAgent, EventorAgent
from electwit.clients import load_model_client
import asyncio
from electwit.utils import (
    load_prompt,
    create_random_background,
    random_number,
    get_closest_response,
    calculate_cosine_similarity,
)

SEED = os.getenv("SEED") or 42

logger = logging.getLogger(__name__)

random.seed(SEED)


class ElecTwit:
    def __init__(
        self,
        voter_models: str = "openrouter/cypher-alpha:free;10",
        candidate_models: str = "openrouter/cypher-alpha:free;2",
        eventer_model: str = "openrouter/cypher-alpha:free",
        candidate_similarity: List[float] = [-1.0, -0.75],
        limits: int = 10,
        memory_limit: int = 5,
    ):
        """
        Initialize the ElecTwit simulation.

        num_people (int): Number of people in the simulation.
        people_models (str): Model names for the people, formatted as "model_name;count,model_name;count,".
        num_candidates (int): Number of candidates in the election.
        candidate_models (str): Model names for the candidates, formatted as "model_name;count,model_name;count,".
        eventer_model (str): Model name for the eventer agent.
        limits (int): Number of posts to keep in the platform's feed.
        memory_limit (int): Number of days consolidation diaries are kept
        """

        self.limits = limits
        self.memory_limit = memory_limit

        self.voter_models = voter_models.split(",")
        self.candidate_models = candidate_models.split(",")

        self.voter_agents = self.initialize_voters(self.voter_models)
        self.candidate_agents = self.initialize_candidates(
            self.candidate_models, similarity=candidate_similarity
        )
        self.candidate_similarity = candidate_similarity
        self.eventer_agent = EventorAgent(
            name="Eventer",
            client=load_model_client(eventer_model, "electwit/prompts/eventer"),
            role="eventer",
            chance_to_act=random_number(0.3, 0.7),
            prompt_dir=os.path.join("electwit", "prompts"),
            limit=memory_limit,
        )
        self.eventer_agent.client.set_system_prompt(
            load_prompt("system_prompt_eventer.txt", "electwit/prompts/eventer")
        )

        self._log_people()
        self._log_candidates()

        self.day = 0
        self.polling_history = []
        self.who_voted_for_who_history = []
        self.platform = Platform()

    def initialize_candidates(
        self, models: List[str], similarity: Optional[List[float]] = None
    ) -> list[ElectionAgent]:
        """
        Initialize agents for the given models.
        """

        while True:
            agents = []
            for model in models:
                model_name, count = model.split(";")

                if not model_name or not count.isdigit():
                    raise ValueError(
                        f"Invalid model format: {model}. Expected 'model_name;count'."
                    )

                count = int(count)
                for _ in range(count):
                    client = load_model_client(
                        model_name, f"electwit/prompts/candidate"
                    )

                    client.set_system_prompt(
                        load_prompt(
                            f"system_prompt_candidate.txt",
                            f"electwit/prompts/candidate",
                        )
                    )

                    background = create_random_background()

                    agents.append(
                        ElectionAgent(
                            name=background["name"],
                            client=client,
                            role="candidate",
                            background=background,
                            chance_to_act=random_number(0.4, 0.9),
                            prompt_dir=os.path.join("electwit", "prompts"),
                            limit=self.memory_limit,
                        )
                    )

            if similarity is None:
                break

            # Check for similarity between candidates
            for i in range(len(agents) - 1):
                sim = calculate_cosine_similarity(
                    agents[i].background, agents[i + 1].background
                )

                logger.info(sim)

                if similarity[0] < sim and sim < similarity[1]:
                    return agents

        return agents

    def initialize_voters(self, models: List[str]) -> list[ElectionAgent]:
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
                client = load_model_client(model_name, "electwit/prompts/voter")

                client.set_system_prompt(
                    load_prompt("system_prompt_voter.txt", "electwit/prompts/voter")
                )

                background = create_random_background()

                agents.append(
                    ElectionAgent(
                        name=background["name"],
                        client=client,
                        role="voter",
                        background=background,
                        chance_to_act=random_number(0.4, 0.9),
                        prompt_dir=os.path.join("electwit", "prompts"),
                        limit=self.memory_limit,
                    )
                )

        return agents

    async def run_polling(self):
        """
        Runs an initial polling to get current preferences
        """

        logger.info("Running initial polling...")

        results = {"ABSTAIN": 0}

        who_voted_for_who = {}

        for candidate in self.candidate_agents:
            results[candidate.name] = 0

        tasks = []
        for agent in self.voter_agents + self.candidate_agents:
            tasks.append(
                asyncio.create_task(
                    agent.act_polling(
                        hour=18,
                        day=self.day,
                        candidates=self._get_candidates_names(),
                        polling_numbers=self._get_most_recent_polling(),
                        current_feed=self.platform.get_feed_as_string(
                            limit=self.limits
                        ),
                        recent_events=self.eventer_agent.get_formatted_events(
                            limit=self.limits
                        ),
                        final_poll=False,
                    )
                )
            )

        votes = await asyncio.gather(*tasks)

        for agent, vote in zip(self.voter_agents + self.candidate_agents, votes):
            action = get_closest_response(
                testing_text=vote["action"],
                responses=["VOTE", "ABSTAIN"],
            )

            if action == "VOTE":
                if not vote.get("candidate"):
                    logger.warning(
                        f"{agent.name} voted without specifying a candidate. Defaulting to abstain."
                    )
                    results["ABSTAIN"] += 1
                    agent.add_journal_entry("Voted without specifying a candidate.")
                    continue

                candidate = get_closest_response(
                    testing_text=vote["candidate"],
                    responses=self._get_candidates_names_list() + ["ABSTAIN"],
                )
                results[candidate] += 1

                who_voted_for_who[agent.name] = candidate

                agent.add_journal_entry(f"Voted for {vote['candidate']}")
            elif action == "ABSTAIN":
                results["ABSTAIN"] += 1

                who_voted_for_who[agent.name] = "ABSTAIN"

                agent.add_journal_entry(f"Abstained from voting")

        logger.info(f"Polling results Day {self.day}: {results}")

        self.polling_history.append(results.copy())

        self.who_voted_for_who_history.append(who_voted_for_who.copy())

    async def final_polling(self) -> dict[str, int]:
        """
        Returns the final polling results as a dictionary.
        """

        logger.info("Running initial final polling...")

        results = {
            "ABSTAIN": 0
        }  # abstaining is not allowed but we just keep it if it happens

        who_voted_for_who = {}

        for candidate in self.candidate_agents:
            results[candidate.name] = 0

        tasks = []
        for agent in self.voter_agents + self.candidate_agents:
            tasks.append(
                asyncio.create_task(
                    agent.act_polling(
                        hour=18,
                        day=self.day,
                        candidates=self._get_candidates_names(),
                        polling_numbers=self._get_most_recent_polling(),
                        current_feed=self.platform.get_feed_as_string(
                            limit=self.limits
                        ),
                        recent_events=self.eventer_agent.get_formatted_events(
                            limit=self.limits
                        ),
                        final_poll=True,
                    )
                )
            )

        votes = await asyncio.gather(*tasks)

        for agent, vote in zip(self.voter_agents + self.candidate_agents, votes):
            action = get_closest_response(
                testing_text=vote["action"],
                responses=["VOTE", "ABSTAIN"],
            )

            if action == "VOTE":
                if not vote.get("candidate"):
                    logger.warning(
                        f"{agent.name}'s FINAL VOTE did NOT specify a candidate. Defaulting to abstain."
                    )
                    agent.add_journal_entry(
                        "Final vote without specifying a candidate."
                    )
                    continue

                candidate = get_closest_response(
                    testing_text=vote["candidate"],
                    responses=self._get_candidates_names_list() + ["ABSTAIN"],
                )

                results[candidate] += 1

                who_voted_for_who[agent.name] = candidate

                agent.add_journal_entry(f"Voted for {vote['candidate']}")
            elif action == "ABSTAIN":
                results["ABSTAIN"] += 1

                who_voted_for_who[agent.name] = "ABSTAIN"

                logger.warning(f"{agent.name}'s FINAL VOTE abstained from voting.")

                agent.add_journal_entry(f"Abstained from final voting")
            else:
                logger.error(
                    f"Unexpected action {vote['action']} from {agent.name} during final polling."
                )
                agent.add_journal_entry(f"Unexpected action: {vote['action']}")

        logger.info(f"Polling results Day {self.day}: {results}")

        self.polling_history.append(results.copy())

        self.who_voted_for_who_history.append(who_voted_for_who.copy())

        return results

    async def run_day(self):
        """
        Simulates a day in the election simulation.
        Each agent has a % chance to interact with the platform.
        """

        self.day += 1

        for hour in range(9, 18):
            logger.info(f"Starting hour {hour} of day {self.day}.")
            # create events
            if self.eventer_agent.chance_to_act > random_number(0, 1):
                await self.eventer_agent.create_event(
                    day=self.day,
                    hour=hour,
                    recent_poll=self._get_most_recent_polling(),
                    event_limit=self.limits,
                )

            combined = self.voter_agents + self.candidate_agents
            random.shuffle(combined)

            async def process_agent(agent):
                await agent.act_posting(
                    hour=hour,
                    day=self.day,
                    candidates=self._get_candidates_names(),
                    polling_numbers=self._get_most_recent_polling(),
                    current_feed=self.platform.get_feed_as_string(limit=self.limits),
                    recent_events=self.eventer_agent.get_formatted_events(
                        limit=self.limits
                    ),
                )
                actions = agent.get_todays_actions()
                self.platform.apply_actions(agent.name, self.day, hour, actions)

            tasks = [
                asyncio.create_task(process_agent(agent))
                for agent in combined
                if random_number(0, 1) < agent.chance_to_act
            ]
            if tasks:
                await asyncio.gather(*tasks)

    async def end_day(self):
        """
        Ends the day in the simulation.

        What it does:
        - Consolidates the diaries of all agents.
        - Runs polling
        """

        await self.run_polling()

        tasks = [
            asyncio.create_task(agent.consolidate_diary(self.day))
            for agent in self.voter_agents + self.candidate_agents
        ]
        await asyncio.gather(*tasks)

        await self.eventer_agent.consolidate_diary(self.day)

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
        return "\n".join(
            [
                f"{candidate} has {votes} votes"
                for candidate, votes in latest_poll.items()
            ]
        )

    async def run_simulation(self, days: int = 7):
        """Runs the entire simulation for a specified number of days."""

        try:
            await self.run_polling()
            for _ in range(days):
                logger.info(f"Starting day {self.day + 1} of the simulation.")
                await self.run_day()
                await self.end_day()

            results = await self.final_polling()

            logger.info("Final polling results:")
            for candidate, votes in results.items():
                logger.info(f"{candidate}: {votes} votes")
        except Exception as e:
            logger.exception(f"An error occurred during the simulation: {e}")
            raise
        finally:
            logger.info("Simulation completed. Saving results...")

            self._export_to_json()

            # log the platform state and agent states and polling

            with open(get_log_dir() / "simulation_results.txt", "w") as f:
                f.write("--- Platform State: ---\n")
                f.write(str(self.platform))
                f.write("\n\n---Polling History:---\n")
                for day, poll in enumerate(self.polling_history, start=1):
                    f.write(f"Day {day}: {poll}\n")

                f.write("\n\n--- Events: ---\n")
                f.write(
                    self.eventer_agent.get_formatted_events(
                        len(self.eventer_agent.events)
                    )
                )

                f.write("\n\n--- People Agents: ---\n")
                for agent in self.voter_agents:
                    f.write(
                        str(agent) + "\n"
                        f"Consolidated Diary: {agent.formatted_consolidated_diary(limit=0)}\n"
                        f"Today's Diary: {agent.formatted_today_diary()}\n"
                    )

                f.write("\n\n--- Candidate Agents: ---\n")
                for agent in self.candidate_agents:
                    f.write(
                        str(agent) + "\n"
                        f"Consolidated Diary: {agent.formatted_consolidated_diary(limit=0)}\n"
                        f"Today's Diary: {agent.formatted_today_diary()}\n"
                    )

                f.write("\n\n--- Eventer Agent: ---\n")
                f.write(
                    str(self.eventer_agent) + "\n"
                    f"Consolidated Diary: {self.eventer_agent.formatted_consolidated_diary(limit=0)}\n"
                    f"Today's Diary: {self.eventer_agent.formatted_today_diary()}\n"
                )

    def _log_people(self):
        logger.info("People in the simulation:")
        for agent in self.voter_agents:
            logger.info(f"Person: {agent.name}, Political View: {agent.background}")

    def _log_candidates(self):
        logger.info("Candidates in the simulation:")
        for agent in self.candidate_agents:
            logger.info(f"Candidate: {agent.name}, Political View: {agent.background}")

    def _export_to_json(self, checkpoint: int = None) -> dict:
        """
        Saves the current state of the simulation to a checkpoint json file.
        """

        file = "checkpoint.json"
        if checkpoint is not None:
            file = f"checkpoint_{checkpoint}.json"

        with open(get_log_dir() / file, "w") as f:
            checkpoint_data = {
                "day": self.day,
                "polling_history": self.polling_history,
                "who_voted_for_who_history": self.who_voted_for_who_history,
                "people_agents": [
                    agent._export_to_json() for agent in self.voter_agents
                ],
                "candidate_agents": [
                    agent._export_to_json() for agent in self.candidate_agents
                ],
                "eventer_agent": self.eventer_agent._export_to_json(),
                "platform": self.platform._export_to_json(),
                "limits": self.limits,
                "memory_limit": self.memory_limit,
                "candidate_similarity": self.candidate_similarity,
            }
            f.write(json.dumps(checkpoint_data, indent=4))
