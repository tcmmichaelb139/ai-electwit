import os
import logging
import random
from faker import Faker
from thefuzz import process  # uses Levenshtein Distance
import numpy as np

from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

SEED = os.getenv("SEED", 42)


random.seed(SEED)
fake = Faker(["en_US"])
fake.seed_instance(SEED)


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


def create_background_prompt(
    name: str,
    economic_policy: int,
    social_authority: int,
    governmental_power: int,
    foreign_policy: int,
    environmental_approach: int,
    national_identity: int,
    openness: int,
    conscientiousness: int,
    extraversion: int,
    agreeableness: int,
    emotional_stability: int,
) -> str:
    """
    Creates a background prompt based on the user info
    """

    background_prompt = load_prompt(
        "stances_background_prompt.txt", "electwit/prompts/background"
    )

    background_prompt = background_prompt.format(
        name=name,
        economic_policy=economic_policy,
        social_authority=social_authority,
        governmental_power=governmental_power,
        foreign_policy=foreign_policy,
        environmental_approach=environmental_approach,
        national_identity=national_identity,
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        emotional_stability=emotional_stability,
    )

    return background_prompt


def create_random_background() -> dict:
    """
    Creates a random background prompt using predefined stance categories.
    """

    name = fake.unique.name()
    economic_policy = random.randint(-100, 100)
    social_authority = random.randint(-100, 100)
    governmental_power = random.randint(-100, 100)
    foreign_policy = random.randint(-100, 100)
    environmental_approach = random.randint(-100, 100)
    national_identity = random.randint(-100, 100)
    openness = random.randint(-100, 100)
    conscientiousness = random.randint(-100, 100)
    extraversion = random.randint(-100, 100)
    agreeableness = random.randint(-100, 100)
    emotional_stability = random.randint(-100, 100)
    return {
        "name": name,
        "economic_policy": economic_policy,
        "social_authority": social_authority,
        "governmental_power": governmental_power,
        "foreign_policy": foreign_policy,
        "environmental_approach": environmental_approach,
        "national_identity": national_identity,
        "openness": openness,
        "conscientiousness": conscientiousness,
        "extraversion": extraversion,
        "agreeableness": agreeableness,
        "emotional_stability": emotional_stability,
    }


def apply_background_prompt(background: dict) -> str:
    """
    Applies the background prompt to create a formatted string.
    """

    if not background:
        return ""

    return create_background_prompt(
        name=background["name"],
        economic_policy=background["economic_policy"],
        social_authority=background["social_authority"],
        governmental_power=background["governmental_power"],
        foreign_policy=background["foreign_policy"],
        environmental_approach=background["environmental_approach"],
        national_identity=background["national_identity"],
        openness=background["openness"],
        conscientiousness=background["conscientiousness"],
        extraversion=background["extraversion"],
        agreeableness=background["agreeableness"],
        emotional_stability=background["emotional_stability"],
    )


def random_number(a: float, b: float) -> float:
    """
    Returns a random float between a and b.
    """

    return random.uniform(a, b)


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


def calculate_cosine_similarity(bg1: dict, bg2: dict) -> float:
    """
    Calculate the cosine similarity between two background dictionaries.
    """
    voter_vector = []
    candidate_vector = []

    political_attrs = [
        "economic_policy",
        "social_authority",
        "governmental_power",
        "foreign_policy",
        "environmental_approach",
        "national_identity",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "emotional_stability",
    ]
    for attr in political_attrs:
        voter_score = bg1.get(attr, 0)  # Default to neutral 0 if missing
        cand_score = bg2.get(attr, 0)
        voter_vector.append(voter_score / 100.0)
        candidate_vector.append(cand_score / 100.0)

    # cosine similarity
    voter_vector = np.array(voter_vector)
    candidate_vector = np.array(candidate_vector)

    dot_product = np.dot(voter_vector, candidate_vector)
    voter_norm = np.linalg.norm(voter_vector)
    candidate_norm = np.linalg.norm(candidate_vector)

    if voter_norm == 0 or candidate_norm == 0:
        return 0

    return dot_product / (voter_norm * candidate_norm)
