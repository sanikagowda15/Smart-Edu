# ============================================================
# SMART EDU - RESOURCE RECOMMENDATION ENGINE
# ONE LINK PER DETECTED TOPIC
# ============================================================

import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESOURCE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "resources.csv"
)


# ============================================================
# TOPIC -> ONE LEARNING RESOURCE LINK
# ============================================================

TOPIC_URLS = {

    "Python":
        "https://www.geeksforgeeks.org/python/python-programming-language-tutorial/",

    "Machine Learning":
        "https://www.geeksforgeeks.org/courses/search?query=machine%20learning",

    "Data Analytics":
        "https://www.geeksforgeeks.org/courses/search?query=data%20analytics",

    "Data Science":
        "https://www.geeksforgeeks.org/courses/search?query=data%20science",

    "Computer Vision":
        "https://www.geeksforgeeks.org/courses/search?query=computer%20vision",

    "Artificial Intelligence":
        "https://www.geeksforgeeks.org/courses/search?query=artificial%20intelligence",

    "Deep Learning":
        "https://www.geeksforgeeks.org/courses/search?query=deep%20learning",

    "NLP":
        "https://www.geeksforgeeks.org/courses/search?query=natural%20language%20processing",

    "Database":
        "https://www.geeksforgeeks.org/courses/search?query=database",

    "Web Development":
        "https://www.geeksforgeeks.org/courses/search?query=web%20development",

    "Java":
        "https://www.geeksforgeeks.org/courses/search?query=java",

    "C++":
        "https://www.geeksforgeeks.org/courses/search?query=c%2B%2B",

    "JavaScript":
        "https://www.geeksforgeeks.org/courses/search?query=javascript"
}


# ============================================================
# LOAD RESOURCES
# ============================================================

def load_resources():

    if not os.path.exists(
        RESOURCE_FILE
    ):

        print(
            "Resource file not found:",
            RESOURCE_FILE
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            RESOURCE_FILE
        )

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        return df

    except Exception as e:

        print(
            "Error loading resources:",
            e
        )

        return pd.DataFrame()


# ============================================================
# PREPARE RESOURCE TEXT
# ============================================================

def prepare_resource_text(df):

    text_columns = [
        "title",
        "subject",
        "level",
        "type",
        "difficulty"
    ]

    for column in text_columns:

        if column not in df.columns:

            df[column] = ""

    df["combined_text"] = (

        df["title"]
        .fillna("")
        .astype(str)

        + " "

        + df["subject"]
        .fillna("")
        .astype(str)

        + " "

        + df["level"]
        .fillna("")
        .astype(str)

        + " "

        + df["type"]
        .fillna("")
        .astype(str)

        + " "

        + df["difficulty"]
        .fillna("")
        .astype(str)
    )

    return df


# ============================================================
# GET ONE TOPIC URL
# ============================================================

def get_topic_url(topic):

    if not topic:

        return None

    topic = str(
        topic
    ).strip()

    return TOPIC_URLS.get(
        topic
    )


# ============================================================
# RECOMMEND ONE RESOURCE LINK
# ============================================================

def recommend_resources(
    topic,
    subject="",
    level="",
    completed_resources=None
):

    # --------------------------------------------------------
    # Load resources
    # --------------------------------------------------------

    df = load_resources()

    if df.empty:

        return df


    # --------------------------------------------------------
    # Prepare text
    # --------------------------------------------------------

    df = prepare_resource_text(
        df
    )


    # --------------------------------------------------------
    # Remove completed resources
    # --------------------------------------------------------

    if completed_resources is None:

        completed_resources = []


    completed_resources = [

        str(resource)
        .strip()
        .lower()

        for resource
        in completed_resources
    ]


    if completed_resources:

        df = df[
            ~df["title"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                completed_resources
            )
        ].copy()


    if df.empty:

        return df


    # --------------------------------------------------------
    # Build query
    # --------------------------------------------------------

    query_parts = []


    if topic:

        query_parts.append(
            str(topic)
        )


    if subject:

        query_parts.append(
            str(subject)
        )


    if level:

        query_parts.append(
            str(level)
        )


    query = " ".join(
        query_parts
    ).strip()


    # --------------------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------------------

    if query:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )


        resource_matrix = (
            vectorizer.fit_transform(
                df["combined_text"]
            )
        )


        query_vector = (
            vectorizer.transform(
                [query]
            )
        )


        similarity_scores = (
            cosine_similarity(
                query_vector,
                resource_matrix
            )[0]
        )


        df["similarity_score"] = (
            similarity_scores
        )

    else:

        df["similarity_score"] = 0


    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    if "rating" in df.columns:

        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        ).fillna(0)

    else:

        df["rating"] = 0


    # --------------------------------------------------------
    # Rank resources
    # --------------------------------------------------------

    df = df.sort_values(

        by=[
            "similarity_score",
            "rating"
        ],

        ascending=[
            False,
            False
        ]
    )


    # --------------------------------------------------------
    # Select ONE resource
    # --------------------------------------------------------

    df = df.head(
        1
    ).copy()


    # --------------------------------------------------------
    # Add ONE learning URL
    # --------------------------------------------------------

    df["url"] = get_topic_url(
        topic
    )


    # --------------------------------------------------------
    # Add similarity percentage
    # --------------------------------------------------------

    df["similarity"] = (
        df["similarity_score"] * 100
    ).round(2)


    # --------------------------------------------------------
    # Remove helper columns
    # --------------------------------------------------------

    df = df.drop(

        columns=[
            "combined_text",
            "similarity_score"
        ],

        errors="ignore"
    )


    # --------------------------------------------------------
    # Return ONE resource
    # --------------------------------------------------------

    return df