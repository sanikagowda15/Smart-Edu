import os
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROGRESS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "progress.csv"
)

RESOURCES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "resources.csv"
)


def load_progress_file():
    """Load progress data or create the file if it does not exist."""

    if not os.path.exists(PROGRESS_FILE):
        data = pd.DataFrame(
            columns=[
                "student_id",
                "student_name",
                "resource",
                "title",
                "status",
                "progress"
            ]
        )

        data.to_csv(
            PROGRESS_FILE,
            index=False
        )

        return data

    return pd.read_csv(PROGRESS_FILE)


def ensure_user_progress(student_name):
    """Ensure the student has a progress record for every resource."""

    student_name = str(student_name).strip()

    progress_data = load_progress_file()
    resources_data = pd.read_csv(RESOURCES_FILE)

    if progress_data.empty:
        progress_data = pd.DataFrame(
            columns=[
                "student_id",
                "student_name",
                "resource",
                "title",
                "status",
                "progress"
            ]
        )

    progress_data["student_name"] = (
        progress_data["student_name"]
        .astype(str)
        .str.strip()
    )

    progress_data["title"] = (
        progress_data["title"]
        .astype(str)
        .str.strip()
    )

    existing_titles = set(
        progress_data[
            progress_data["student_name"].str.lower()
            == student_name.lower()
        ]["title"].tolist()
    )

    new_records = []

    for _, resource in resources_data.iterrows():

        title = str(
            resource["title"]
        ).strip()

        if title not in existing_titles:

            new_records.append({
                "student_id": student_name,
                "student_name": student_name,
                "resource": title,
                "title": title,
                "status": "Not Started",
                "progress": 0
            })

    if new_records:

        new_data = pd.DataFrame(
            new_records
        )

        progress_data = pd.concat(
            [
                progress_data,
                new_data
            ],
            ignore_index=True
        )

        progress_data.to_csv(
            PROGRESS_FILE,
            index=False
        )


def create_user_progress(student_name):
    """Create progress records for a newly registered student."""

    ensure_user_progress(student_name)


def get_progress(student_name):
    """Return progress statistics for the student."""

    student_name = str(
        student_name
    ).strip()

    ensure_user_progress(
        student_name
    )

    data = load_progress_file()

    data["student_name"] = (
        data["student_name"]
        .astype(str)
        .str.strip()
    )

    user_data = data[
        data["student_name"].str.lower()
        == student_name.lower()
    ]

    if user_data.empty:
        return {
            "total_resources": 0,
            "completed": 0,
            "in_progress": 0,
            "not_started": 0,
            "average_progress": 0
        }

    total_resources = len(user_data)

    completed = len(
        user_data[
            user_data["status"].astype(str).str.strip()
            == "Completed"
        ]
    )

    in_progress = len(
        user_data[
            user_data["status"].astype(str).str.strip()
            == "In Progress"
        ]
    )

    not_started = len(
        user_data[
            user_data["status"].astype(str).str.strip()
            == "Not Started"
        ]
    )

    average_progress = round(
        pd.to_numeric(
            user_data["progress"],
            errors="coerce"
        ).fillna(0).mean(),
        2
    )

    return {
        "total_resources": total_resources,
        "completed": completed,
        "in_progress": in_progress,
        "not_started": not_started,
        "average_progress": average_progress
    }


def start_resource(student_name, resource_title):
    """
    Mark a resource as In Progress.

    Starting a resource does not automatically
    assign 50% progress.
    """

    student_name = str(
        student_name
    ).strip()

    resource_title = str(
        resource_title
    ).strip()

    ensure_user_progress(
        student_name
    )

    data = load_progress_file()

    data["student_name"] = (
        data["student_name"]
        .astype(str)
        .str.strip()
    )

    data["title"] = (
        data["title"]
        .astype(str)
        .str.strip()
    )

    condition = (
        data["student_name"].str.lower()
        == student_name.lower()
    ) & (
        data["title"].str.lower()
        == resource_title.lower()
    )

    if condition.any():

        current_status = str(
            data.loc[
                condition,
                "status"
            ].iloc[0]
        ).strip()

        if current_status != "Completed":

            data.loc[
                condition,
                "status"
            ] = "In Progress"

            data.loc[
                condition,
                "progress"
            ] = 0

            data.to_csv(
                PROGRESS_FILE,
                index=False
            )


def complete_resource(student_name, resource_title):
    """Mark a resource as completed with 100% progress."""

    student_name = str(
        student_name
    ).strip()

    resource_title = str(
        resource_title
    ).strip()

    ensure_user_progress(
        student_name
    )

    data = load_progress_file()

    data["student_name"] = (
        data["student_name"]
        .astype(str)
        .str.strip()
    )

    data["title"] = (
        data["title"]
        .astype(str)
        .str.strip()
    )

    condition = (
        data["student_name"].str.lower()
        == student_name.lower()
    ) & (
        data["title"].str.lower()
        == resource_title.lower()
    )

    if condition.any():

        data.loc[
            condition,
            "status"
        ] = "Completed"

        data.loc[
            condition,
            "progress"
        ] = 100

        data.to_csv(
            PROGRESS_FILE,
            index=False
        )