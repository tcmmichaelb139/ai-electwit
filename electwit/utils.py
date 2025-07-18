import os
import logging
import random
from faker import Faker
from thefuzz import process  # uses Levenshtein Distance

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
    age: int,
    gender: str,
    race: str,
    family_status: str,
    household_income: int,
    religious_stance: str,
    education_level: str,
    community_type: str,
    employment_sector: str,
    economic_regulation_stance: int,
    social_program_stance: int,
    social_progressivism_stance: int,
    nationalism_vs_globalism_stance: int,
    authority_stance: int,
    environmental_priorities_stance: int,
) -> str:
    """
    Creates a background prompt based on the user info
    """

    background_prompt = load_prompt(
        "generic_background_prompt.txt", "electwit/prompts/background"
    )

    background_prompt = background_prompt.format(
        name=name,
        age=age,
        gender=gender,
        race=race,
        family_status=family_status,
        household_income=household_income,
        religious_stance=religious_stance,
        education_level=education_level,
        community_type=community_type,
        employment_sector=employment_sector,
        economic_regulation_stance=economic_regulation_stance,
        social_program_stance=social_program_stance,
        social_progressivism_stance=social_progressivism_stance,
        nationalism_vs_globalism_stance=nationalism_vs_globalism_stance,
        authority_stance=authority_stance,
        environmental_priorities_stance=environmental_priorities_stance,
    )

    return background_prompt


def create_random_background() -> dict:
    """
    Creates a random background prompt using Faker and predefined categories.
    """

    gender = random.choice(["male", "female", "non-binary"])
    name = ""
    if gender == "male":
        name = fake.unique.name_male()
    elif gender == "female":
        name = fake.unique.name_female()
    else:
        name = fake.unique.name_nonbinary()
    age = random.randint(18, 75)
    race = random.choice(
        [
            "White",
            "Black or African American",
            "Asian",
            "Hispanic or Latino",
            "Native American",
            "Pacific Islander",
        ]
    )
    family_status = random.choice(
        [
            "Single",
            "Married",
            "Divorced",
            "Widowed",
            "Cohabitating",
        ]
    )
    household_income = random.randint(20000, 200000)
    religious_stance = random.choice(
        [
            "Atheist",
            "Agnostic",
            "Spiritual but not religious",
            "Casually religious",
            "Devoutly religious",
        ]
    )
    education_level = random.choice(
        [
            "High School",
            "Trade School",
            "Community College",
            "Bachelor's Degree",
            "Master's Degree",
            "Doctorate",
        ]
    )
    community_type = random.choice(
        [
            "Urban",
            "Suburban",
            "Rural",
            "Small Town",
            "Metropolitan Area",
            "Tech Hub",
        ]
    )
    employment_sector = random.choice(
        [
            "Not Employed",
            "Manufacturing",
            "Service Industry",
            "Technology",
            "Healthcare",
            "Education",
            "Finance",
            "Government",
            "Non-Profit",
            "Business Owner",
        ]
    )
    economic_regulation_stance = random.randint(0, 100)
    social_program_stance = random.randint(0, 100)
    social_progressivism_stance = random.randint(0, 100)
    nationalism_vs_globalism_stance = random.randint(0, 100)
    authority_stance = random.randint(0, 100)
    environmental_priorities_stance = random.randint(0, 100)
    return {
        "name": name,
        "age": age,
        "gender": gender,
        "race": race,
        "family_status": family_status,
        "household_income": household_income,
        "religious_stance": religious_stance,
        "education_level": education_level,
        "community_type": community_type,
        "employment_sector": employment_sector,
        "economic_regulation_stance": economic_regulation_stance,
        "social_program_stance": social_program_stance,
        "social_progressivism_stance": social_progressivism_stance,
        "nationalism_vs_globalism_stance": nationalism_vs_globalism_stance,
        "authority_stance": authority_stance,
        "environmental_priorities_stance": environmental_priorities_stance,
    }


def apply_background_prompt(background: dict) -> str:
    """
    Applies the background prompt to create a formatted string.
    """

    if not background:
        return ""

    return create_background_prompt(
        name=background["name"],
        age=background["age"],
        gender=background["gender"],
        race=background["race"],
        family_status=background["family_status"],
        household_income=background["household_income"],
        religious_stance=background["religious_stance"],
        education_level=background["education_level"],
        community_type=background["community_type"],
        employment_sector=background["employment_sector"],
        economic_regulation_stance=background["economic_regulation_stance"],
        social_program_stance=background["social_program_stance"],
        social_progressivism_stance=background["social_progressivism_stance"],
        nationalism_vs_globalism_stance=background["nationalism_vs_globalism_stance"],
        authority_stance=background["authority_stance"],
        environmental_priorities_stance=background["environmental_priorities_stance"],
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
