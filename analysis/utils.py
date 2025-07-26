import json

FILES = [
    "../logs/2025-07-21_10-56-37/checkpoint.json",
    "../logs/2025-07-21_13-46-49/checkpoint.json",
    "../logs/2025-07-21_16-26-25/checkpoint.json",
    "../logs/2025-07-21_18-17-49/checkpoint.json",
    "../logs/2025-07-22_10-29-24/checkpoint.json",
    "../logs/2025-07-22_12-15-49/checkpoint.json",
]

# this is for multiple of the same tests (gemini-2.5-flash vs openai 4.1 mini)
FILES2 = [
    "../logs/2025-07-21_16-26-25/checkpoint.json",
    "../logs/2025-07-23_20-51-13/checkpoint.json",
    "../logs/2025-07-23_22-58-29/checkpoint.json",
    "../logs/2025-07-24_10-25-20/checkpoint.json",
    "../logs/2025-07-24_13-09-33/checkpoint.json",
    "../logs/2025-07-25_10-08-56/checkpoint.json",
]


def get_data(type_data: str = "all"):
    """gets the data from the files
    type_data
    all_data - gets all data from all files
    same_seed - FILES (which has the same seed, but different models)
    different_seed - FILES2 (which has different seeds, but the same models)
    """
    if type_data == "all_data":
        files = list(set(FILES + FILES2))
    elif type_data == "same_seed":
        files = FILES
    elif type_data == "different_seed":
        files = FILES2
    else:
        raise ValueError(
            "Invalid type_data. Use 'all_data', 'same_seed', or 'different_seed'."
        )

    data = []
    for file in files:
        with open(file + "-updated.json", "r") as f:
            data.append(json.load(f))

    return data


from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


def create_custom_color_palette(n_colors):
    """Create a custom color palette with colors similar to the requested palette"""

    custom_colors = [
        "#d6604d",
        "#4393c3",
        "#5aae61",
        "#8073ac",
        "#fee090",
        "#de77ae",
        "#bf812d",
        "#2c7fb8",
        "#c994c7",
        "#41b6c4",
        "#a1dab4",
        "#878787",
        "#df65b0",
        "#969696",
        "#fd8d3c",
        "#525252",
    ]

    if n_colors <= len(custom_colors):
        indices = range(n_colors)
        return [custom_colors[i] for i in indices]
    else:
        extra_needed = n_colors - len(custom_colors)

        for i in range(extra_needed):
            color1 = mcolors.to_rgb(custom_colors[i % len(custom_colors)])
            color2 = mcolors.to_rgb(custom_colors[(i + 1) % len(custom_colors)])
            new_color = [(c1 + c2) / 2 for c1, c2 in zip(color1, color2)]
            custom_colors.append(mcolors.to_hex(new_color))

        return custom_colors


def calculate_similarity(voter_profile, candidate_profile):
    """Calculate cosine similarity between a voter and candidate based on their background profiles."""
    import numpy as np

    voter_bg = voter_profile.get("background", {})
    candidate_bg = candidate_profile.get("background", {})

    if not voter_bg or not candidate_bg:
        return 0

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
        voter_score = voter_bg.get(attr, 0)  # Default to neutral 0 if missing
        cand_score = candidate_bg.get(attr, 0)
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
