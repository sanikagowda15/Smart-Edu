import pandas as pd


# Load educational resource dataset
data = pd.read_csv("data/resources.csv")


def get_analytics():
    # Number of resources by subject
    subject_counts = (
        data["subject"]
        .value_counts()
        .to_dict()
    )

    # Number of resources by learning level
    level_counts = (
        data["level"]
        .value_counts()
        .to_dict()
    )

    # Number of resources by type
    type_counts = (
        data["type"]
        .value_counts()
        .to_dict()
    )

    # Average resource rating
    average_rating = round(
        data["rating"].mean(),
        2
    )

    # Total number of resources
    total_resources = len(data)

    return {
        "subject_counts": subject_counts,
        "level_counts": level_counts,
        "type_counts": type_counts,
        "average_rating": average_rating,
        "total_resources": total_resources
    }