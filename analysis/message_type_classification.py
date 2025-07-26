import os
import json
import logging
import json_repair
from typing import List, Optional
import re
import asyncio
from rich import print

from thefuzz import process  # uses Levenshtein Distance

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_format = (
    "%(asctime)s [%(levelname)8s] [%(name)s] %(message)s (%(filename)s:%(lineno)s)"
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(console_handler)


def load_prompt(filename: str, prompt_dir: Optional[str] = None) -> str:
    """
    Loads a prompt from a file.
    ---------------------------

    1. If `filename` is an absolute path, it uses that directly.
    2. If `filename` has a directory, it uses it directly.
    3. If `prompt_dir` is provided, it combines `prompt_dir` with `filename`.
    4. If none of the above, it defaults to a "prompts" directory relative to the current file's directory.
    """
    if os.path.isabs(filename):
        prompt_path = filename
    elif os.path.dirname(filename):
        prompt_path = filename
    elif prompt_dir:
        prompt_path = os.path.join(prompt_dir, filename)
    else:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.warning(
            f"Prompt file '{filename}' not found in directory '{prompt_dir}'."
        )

        return ""


class GeminiClient:
    """
    Client interface for Google Gemini API
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.system_prompt = system_prompt

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Generates a response from the Gemini API using the specified model.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )

            if not response.choices:
                logger.warning(f"[{self.model_name}] No choices returned in response.")
                return ""

            content = response.choices[0].message.content.strip()

            if not content:
                logger.warning(
                    f"[{self.model_name}] Empty content returned in response."
                )
                return ""

            return content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise e

    async def generate_response_json_list(self, prompt: str) -> Optional[List[dict]]:
        """
        Returns the response in JSON format
        """
        response = await self.generate_response(prompt)

        response = response.strip()

        if not response:
            logger.warning(f"[{self.model_name}] Empty response generated.")
            return []

        response_json = json_repair.loads(response)

        if not isinstance(response_json, list):
            logger.warning(
                f"[{self.model_name}] Response is not a list: {response_json}. "
            )
            return [response_json]

        return response_json


def _comment_thread_string(comment, indent: str = "  ") -> str:
    """
    Recursively formats a comment and its replies into a string.

    Args:
    - comment (Comment): The comment to format.
    - indent (str): The indentation string for nested replies.
    """
    result = f"{indent}Reply from {comment['name']} (ID: {comment['id']}) (Day: {comment['day']} Hour: {comment['hour']}): {comment['content']}\n(Likes: {comment['likes']})\n"
    ids = f"{comment['id']}\n"
    for reply in comment["replies"]:
        r, i = _comment_thread_string(reply, indent + "  ")
        result += r
        ids += i
    return result, ids


def get_feed_as_string(platform, start, end) -> str:
    feed = platform[start:end].copy()

    result = ""

    ids = ""

    for post in feed:
        result += f"Post from {post['name']} (ID: {post['id']}) (Day: {post['day']} Hour: {post['hour']}):\n{post['content']}\n(Likes: {post['likes']})\n"
        ids += f"{post['id']}\n"
        for comment in post["replies"]:
            r, i = _comment_thread_string(comment, indent="  ")
            result += r
            ids += i

    if not result:
        return "Feed is empty."

    return result.strip(), ids.strip()


def get_closest_response(
    testing_text: str, responses: list[str], threshold: int = 75
) -> str:
    """uses thefuzz"""

    closest = process.extractOne(testing_text, responses)

    if closest[1] < threshold:
        # just to make sure the response has a good match
        logger.error(
            f"Closest match for '{testing_text}' is '{closest[0]}' with a score of {closest[1]}, which is below the threshold."
        )
    elif closest[1] != 100:
        logger.warning(
            f"Closest match for '{testing_text}' is '{closest[0]}' with a score of {closest[1]}, which is not perfect."
        )

    return closest[0]


ALL_TECHNIQUES = [
    "Appeal to Logic",
    "Appeal to Emotion",
    "Appeal to Credibility",
    "Shifting the Burden of Proof",
    "Bandwagon Effect",
    "Distraction",
    "Gaslighting",
    "Appeal to Urgency",
    "Deception",
    "Lying",
    "Feigning Ignorance",
    "Vagueness",
    "Minimization",
    "Self-Deprecation",
    "Projection",
    "Appeal to Relationship",
    "Humor",
    "Sarcasm",
    "Withholding Information",
    "Exaggeration",
    "Denial without Evidence",
    "Strategic Voting Suggestion",
    "Appeal to Rules",
    "Confirmation Bias Exploitation",
    "Information Overload",
]


def get_closest_techniques(techniques):
    """return the closest techniques from ALL_TECHNIQUES"""

    closest_techniques = []
    for technique in techniques:
        closest = get_closest_response(technique, ALL_TECHNIQUES)
        closest_techniques.append(closest)
    return closest_techniques


def add_classification_to_json(platform, classification_response: List[dict]) -> None:
    if not classification_response:
        logger.warning("Classification response is empty")
        return

    classification_lookup = {
        item["ID"]: item["techniques"] for item in classification_response
    }
    used_ids = set()

    def _add_classification_recursive(items: List[dict]) -> None:
        """Recursively process posts and comments"""
        for item in items:
            item_id = item.get("id")
            if item_id and item_id in classification_lookup:
                item["techniques"] = get_closest_techniques(
                    classification_lookup[item_id]
                )
                used_ids.add(item_id)
                logger.debug(f"Added classification to item ID: {item_id}")

            if "replies" in item and item["replies"]:
                _add_classification_recursive(item["replies"])

    posts = platform
    _add_classification_recursive(posts)

    all_classification_ids = set(classification_lookup.keys())
    unused_ids = all_classification_ids - used_ids

    if unused_ids:
        for unused_id in unused_ids:
            logger.error(
                f"Classification ID '{unused_id}' not found in any post or comment"
            )

    logger.info(f"Successfully added classifications to {len(used_ids)} items")
    logger.info(f"Failed to match {len(unused_ids)} classification IDs")


def get_posts_with_classifications(
    platform, start_index: int = 0, end_index: int = None
) -> List[dict]:
    posts = platform

    if end_index is None:
        end_index = len(posts)

    start_index = max(0, start_index)
    end_index = min(len(posts), end_index)

    selected_posts = posts[start_index:end_index]
    classified_posts = []

    def _collect_classified_items(items: List[dict]) -> None:
        """Recursively collect items with classifications"""
        for item in items:
            if "techniques" in item:
                classified_posts.append(item)

            if "replies" in item and item["replies"]:
                _collect_classified_items(item["replies"])

    _collect_classified_items(selected_posts)
    return classified_posts


def number_of_messages(platform: List[dict]) -> int:
    def _recursive_count(item) -> int:
        """
        Recursively counts the number of messages in posts and comments.
        """
        count = 1
        for reply in item.get("replies", []):
            count += _recursive_count(reply)
        return count

    total_messages = 0

    for post in platform:
        total_messages += _recursive_count(post)

    return total_messages


SYSTEM_PROMPT = load_prompt(
    "message_type_classification.txt", prompt_dir="analysis/prompts"
)

################################# Change this #################################
all_data_files = [
    # main
    # "logs/2025-07-21_10-56-37/checkpoint.json",
    # "logs/2025-07-21_13-46-49/checkpoint.json",
    # "logs/2025-07-21_16-26-25/checkpoint.json",
    # "logs/2025-07-21_18-17-49/checkpoint.json",
    # "logs/2025-07-22_10-29-24/checkpoint.json",
    # "logs/2025-07-22_12-15-49/checkpoint.json",
    # other stuff
    # "logs/2025-07-23_20-51-13/checkpoint.json",
    # "logs/2025-07-23_22-58-29/checkpoint.json",
    # "logs/2025-07-24_10-25-20/checkpoint.json",
    # "logs/2025-07-24_13-09-33/checkpoint.json",
    "logs/2025-07-25_10-08-56/checkpoint.json",
]
group_sizing = 50


async def main(platform, group_size: int = 10):
    # Create all the tasks for concurrent execution
    tasks = []
    bounds_list = []

    for lower_bound in range(0, len(platform), group_size):
        upper_bound = min(lower_bound + group_size, len(platform))
        bounds_list.append((lower_bound, upper_bound))

        # Create async task for each batch
        tasks.append(
            asyncio.create_task(
                get_techniques_used(
                    *get_feed_as_string(platform, lower_bound, upper_bound),
                    lower_bound,
                )
            )
        )

    logger.info(f"Starting concurrent classification of {len(tasks)} batches")

    try:
        # Run all tasks concurrently
        all_responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_classifications = 0
        total_classified_items = 0

        for i, (response, (lower_bound, upper_bound)) in enumerate(
            zip(all_responses, bounds_list)
        ):
            logger.info(
                f"Processing batch {i+1}/{len(tasks)}: posts {lower_bound} to {upper_bound}"
            )

            if isinstance(response, Exception):
                logger.error(f"Error in batch {i+1}: {response}")
                continue

            if response:
                # Apply the classification to the JSON data
                add_classification_to_json(platform, response)

                classified_items = get_posts_with_classifications(
                    platform, lower_bound, upper_bound
                )
                batch_count = len(classified_items)
                total_classified_items += batch_count
                successful_classifications += 1

                total_messages = number_of_messages(platform[lower_bound:upper_bound])

                logger.info(
                    f"Batch {i+1}: Found {batch_count} items with classifications out of {total_messages} messages"
                )
                if batch_count != total_messages:
                    logger.error(f"MISMATCHED COUNT: {batch_count} != {total_messages}")

                # Print summary of classifications for first few items
                for item in classified_items[:2]:  # Show first 2 as example
                    print(f"ID: {item['id']}, Techniques: {item.get('techniques', [])}")
            else:
                logger.warning(f"Empty response for batch {i+1}")

        logger.info(
            f"Completed processing: {successful_classifications}/{len(tasks)} batches successful"
        )
        logger.info(f"Total items classified: {total_classified_items}")

    except Exception as e:
        logger.error(f"Error during concurrent processing: {e}")


for data_file in all_data_files:
    json_df = json.load(open(data_file, "r"))

    logger.info("Initialized GeminiClient and loaded data")

    client = GeminiClient(
        model_name="gemini-2.5-pro",
        system_prompt=SYSTEM_PROMPT,
    )

    platform = json_df["platform"]["platform"]

    async def get_techniques_used(feed_str: str, id_str: str, id_: int):
        logger.info(f"[BATCH {id_}] Starting classification of techniques")

        response = await client.generate_response_json_list(
            f"""Classify the following posts and comments:\n{feed_str}\n\nThe following are all the IDs in the feed:\n{id_str}"""
        )

        logger.info(
            f"[BATCH {id_}] Completed classification - Responses: {len(response)}"
        )
        return response

    asyncio.run(main(platform, group_sizing))

    json_df["platform"]["platform"] = platform

    with open(data_file + "-updated.json", "w") as f:
        json.dump(json_df, f, indent=4)
