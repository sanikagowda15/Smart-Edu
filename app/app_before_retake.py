
import sys
import os
from datetime import datetime

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

# ============================================================
# FLASK IMPORTS
# ============================================================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

# ============================================================
# OTHER IMPORTS
# ============================================================

import pandas as pd

# ============================================================
# PROJECT MODULES
# ============================================================

from models.recommendation import recommend_resources
from models.analytics import get_analytics

from models.progress import (
    get_progress,
    create_user_progress,
    start_resource,
    complete_resource
)

from models.vision import analyze_image
from models.answer_engine import generate_answer

from models.auth import (
    register_user,
    login_user,
    get_registration_error
)

from models.resource_details import RESOURCE_DETAILS
from models.learning_content import LEARNING_CONTENT

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

app.secret_key = "smart-edu-demo-secret-key"

# ============================================================
# LOAD RESOURCES DATASET
# ============================================================

RESOURCES_FILE = os.path.join(
    BASE_DIR, "data", "resources.csv"
)

data = pd.read_csv(RESOURCES_FILE)

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    progress_data = get_progress(session["user"])

    return render_template(
        "dashboard.html",
        user_name=session["user"],
        total_resources=progress_data["total_resources"],
        completed=progress_data["completed"],
        in_progress=progress_data["in_progress"],
        not_started=progress_data["not_started"],
        average_progress=progress_data["average_progress"]
    )

# ============================================================
# RESOURCES PAGE
# ============================================================

@app.route("/resources")
def resources():
    resource_list = data.to_dict(orient="records")

    return render_template(
        "resources.html",
        resources=resource_list
    )

# ============================================================
# RESOURCE DETAILS
# ============================================================

@app.route("/resource/<int:resource_id>")
def resource_details(resource_id):
    resource = data[data["resource_id"] == resource_id]

    if resource.empty:
        return "Resource not found", 404

    resource_data = resource.iloc[0].to_dict()

    extra_details = RESOURCE_DETAILS.get(
        resource_data["title"],
        {
            "description": (
                "Educational resource for learning this topic."
            ),
            "topics": [],
            "objectives": [],
            "external_links": []
        }
    )

    resource_data.update(extra_details)

    return render_template(
        "resource_details.html",
        resource=resource_data
    )

# ============================================================
# START LEARNING
# ============================================================

@app.route("/learn/<int:resource_id>")
def learn(resource_id):
    if "user" not in session:
        return redirect(url_for("login"))

    resource = data[data["resource_id"] == resource_id]

    if resource.empty:
        return "Resource not found", 404

    resource_data = resource.iloc[0].to_dict()

    extra_details = RESOURCE_DETAILS.get(
        resource_data["title"],
        {
            "description": (
                "Educational resource for learning this topic."
            ),
            "topics": [],
            "objectives": [],
            "external_links": []
        }
    )

    resource_data.update(extra_details)

    start_resource(
        session["user"],
        resource_data["title"]
    )

    learning_data = LEARNING_CONTENT.get(
        resource_data["title"]
    )

    if learning_data is None:
        learning_data = {
            "introduction": (
                "Learning content is being prepared for this resource."
            ),
            "content": []
        }

    return render_template(
        "learn.html",
        resource=resource_data,
        learning=learning_data
    )

# ============================================================
# COMPLETE LEARNING
# ============================================================

@app.route("/complete/<int:resource_id>", methods=["POST"])
def complete_learning(resource_id):
    if "user" not in session:
        return redirect(url_for("login"))

    resource = data[data["resource_id"] == resource_id]

    if resource.empty:
        return "Resource not found", 404

    resource_data = resource.iloc[0].to_dict()

    complete_resource(
        session["user"],
        resource_data["title"]
    )

    return redirect(url_for("progress"))

# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    recommendations_list = None

    if request.method == "POST":
        subject = request.form.get("subject", "")
        level = request.form.get("level", "")
        resource_type = request.form.get("resource_type", "")

        completed_resources = []

        if "user" in session:
            from models.progress import load_progress_file

            progress_file = load_progress_file()

            user_progress = progress_file[
                progress_file["student_name"]
                .astype(str)
                .str.strip()
                .str.lower()
                == session["user"].strip().lower()
            ]

            completed_resources = user_progress[
                user_progress["status"]
                .astype(str)
                .str.strip()
                == "Completed"
            ]["title"].tolist()

        recommendations_list = recommend_resources(
            subject,
            level,
            resource_type,
            completed_resources
        ).to_dict(orient="records")

    return render_template(
        "recommendations.html",
        recommendations=recommendations_list
    )

# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics")
def analytics():
    analytics_data = get_analytics()

    return render_template(
        "analytics.html",
        subject_counts=analytics_data["subject_counts"],
        level_counts=analytics_data["level_counts"],
        type_counts=analytics_data["type_counts"],
        average_rating=analytics_data["average_rating"],
        total_resources=analytics_data["total_resources"]
    )

# ============================================================
# PROGRESS
# ============================================================

@app.route("/progress")
def progress():
    if "user" not in session:
        return redirect(url_for("login"))

    progress_data = get_progress(session["user"])

    return render_template(
        "progress.html",
        total_resources=progress_data["total_resources"],
        completed=progress_data["completed"],
        in_progress=progress_data["in_progress"],
        not_started=progress_data["not_started"],
        average_progress=progress_data["average_progress"]
    )

# ============================================================
# COMPUTER VISION / OCR + ANSWER + RECOMMENDATION
# ============================================================

@app.route("/vision", methods=["GET", "POST"])
def vision():
    result = None
    answer = None
    recommendations_list = []
    message = None

    if request.method == "POST":
        image = request.files.get("image")

        if image is None or image.filename == "":
            return render_template(
                "vision.html",
                result=None,
                answer=None,
                recommendations=[],
                message="Please select an image."
            )

        image_path = os.path.join(
            BASE_DIR, "data", "uploaded_image.jpg"
        )

        image.save(image_path)

        try:
            result = analyze_image(image_path)
        except Exception as e:
            print("Vision error:", e)

            return render_template(
                "vision.html",
                result=None,
                answer=None,
                recommendations=[],
                message=(
                    "Image processing failed. "
                    "Check the image and Tesseract configuration."
                )
            )

        extracted_text = result.get("extracted_text", "")
        detected_topic = result.get("detected_topic", "General")

        if not extracted_text.strip():
            return render_template(
                "vision.html",
                result=result,
                answer=None,
                recommendations=[],
                message=(
                    "No readable text was detected. "
                    "Please upload a clearer image."
                )
            )

        answer = generate_answer(
            extracted_text,
            detected_topic
        )

        completed_resources = []

        if "user" in session:
            try:
                from models.progress import load_progress_file

                progress_file = load_progress_file()

                user_progress = progress_file[
                    progress_file["student_name"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == session["user"].strip().lower()
                ]

                completed_resources = user_progress[
                    user_progress["status"]
                    .astype(str)
                    .str.strip()
                    == "Completed"
                ]["title"].tolist()

            except Exception as e:
                print("Progress loading error:", e)

        if detected_topic != "General":
            try:
                recommendations_list = recommend_resources(
                    detected_topic,
                    "",
                    "",
                    completed_resources
                ).to_dict(orient="records")

            except Exception as e:
                print("Recommendation error:", e)
                recommendations_list = []

    return render_template(
        "vision.html",
        result=result,
        answer=answer,
        recommendations=recommendations_list,
        message=message
    )

# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    message = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        message = get_registration_error(
            name,
            email,
            password
        )

        if message is None:
            success = register_user(
                name,
                email,
                password
            )

            if success:
                create_user_progress(name)
                return redirect(url_for("login"))

            message = (
                "Registration failed. Please try again."
            )

    return render_template(
        "register.html",
        message=message
    )

# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    message = None

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user_name = login_user(email, password)

        if user_name:
            session["user"] = user_name
            create_user_progress(user_name)

            return redirect(url_for("dashboard"))

        message = "Invalid email or password."

    return render_template(
        "login.html",
        message=message
    )

# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

# ============================================================
# SMART QUIZ + WEAK TOPIC DATA COLLECTION
# ============================================================

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "user" not in session:
        return redirect(url_for("login"))

    # Each question has a topic label for performance analysis.
    questions = [
        {
            "question": "Which language is commonly used for AI?",
            "options": ["HTML", "Python", "CSS", "SQL"],
            "answer": "Python",
            "topic": "Python Programming"
        },
        {
            "question": "What does HTML stand for?",
            "options": [
                "Hyper Text Markup Language",
                "High Transfer Machine Language",
                "Hyper Tool Multi Language",
                "Home Text Making Language"
            ],
            "answer": "Hyper Text Markup Language",
            "topic": "Web Development"
        },
        {
            "question": "Which data structure follows FIFO?",
            "options": ["Stack", "Queue", "Tree", "Graph"],
            "answer": "Queue",
            "topic": "Data Structures"
        },
        {
            "question": (
                "Which Python library is commonly used for data analysis?"
            ),
            "options": ["Pandas", "Flask", "Turtle", "Pygame"],
            "answer": "Pandas",
            "topic": "Data Analysis"
        },
        {
            "question": "What does CPU stand for?",
            "options": [
                "Central Processing Unit",
                "Computer Personal Unit",
                "Central Program Utility",
                "Control Processing User"
            ],
            "answer": "Central Processing Unit",
            "topic": "Computer Fundamentals"
        }
    ]

    if request.method == "POST":
        score = 0
        question_results = []

        attempted_at = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Evaluate answers and prepare topic-level records.
        for i, question in enumerate(questions):
            selected = request.form.get(
                f"question_{i}", ""
            )

            is_correct = selected == question["answer"]

            if is_correct:
                score += 1

            question_results.append({
                "student": session["user"],
                "question_index": i + 1,
                "question": question["question"],
                "topic": question["topic"],
                "selected_answer": selected,
                "correct_answer": question["answer"],
                "is_correct": is_correct,
                "attempted_at": attempted_at
            })

        # ----------------------------------------------------
        # SAVE QUESTION-LEVEL RESULTS TO A NEW CSV FILE
        # ----------------------------------------------------

        question_history_file = os.path.join(
            BASE_DIR,
            "data",
            "quiz_question_history.csv"
        )

        try:
            os.makedirs(
                os.path.dirname(question_history_file),
                exist_ok=True
            )

            question_history_df = pd.DataFrame(
                question_results
            )

            write_question_header = (
                not os.path.exists(question_history_file)
                or os.path.getsize(question_history_file) == 0
            )

            question_history_df.to_csv(
                question_history_file,
                mode="a",
                header=write_question_header,
                index=False
            )

        except (OSError, PermissionError) as e:
            print(
                "Question-level quiz history save error:",
                e
            )

        # ----------------------------------------------------
        # SAVE QUIZ ATTEMPT TO THE EXISTING HISTORY FILE
        # ----------------------------------------------------

        history_file = os.path.join(
            BASE_DIR, "data", "quiz_history.csv"
        )

        os.makedirs(
            os.path.dirname(history_file),
            exist_ok=True
        )

        total_questions = len(questions)

        percentage = round(
            (score / total_questions) * 100,
            2
        )

        attempt = {
            "student": session["user"],
            "score": score,
            "total_questions": total_questions,
            "percentage": percentage,
            "attempted_at": attempted_at
        }

        try:
            write_header = (
                not os.path.exists(history_file)
                or os.path.getsize(history_file) == 0
            )

            pd.DataFrame([attempt]).to_csv(
                history_file,
                mode="a",
                header=write_header,
                index=False
            )

        except (OSError, PermissionError) as e:
            print("Quiz history save error:", e)

        # Preserve the existing redirect to Quiz History.
        return redirect(url_for("quiz_history"))

    # Display the quiz normally.
    return render_template(
        "quiz.html",
        questions=questions,
        score=None
    )

# ============================================================
# QUIZ HISTORY PAGE
# ============================================================

@app.route("/quiz-history")
def quiz_history():
    if "user" not in session:
        return redirect(url_for("login"))

    history_file = os.path.join(
        BASE_DIR, "data", "quiz_history.csv"
    )

    history = []

    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file)

            required_columns = {
                "student",
                "score",
                "total_questions",
                "percentage",
                "attempted_at"
            }

            if required_columns.issubset(history_df.columns):
                history_df = history_df[
                    history_df["student"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == session["user"].strip().lower()
                ].copy()

                # Newest attempt first.
                history_df = history_df.iloc[::-1]

                history = history_df.to_dict(
                    orient="records"
                )

        except (pd.errors.EmptyDataError, OSError) as e:
            print("Quiz history read error:", e)
            history = []

    total_attempts = len(history)

    if total_attempts:
        latest_score = (
            f"{history[0]['score']}/"
            f"{history[0]['total_questions']}"
        )

        average_percentage = round(
            sum(
                float(item["percentage"])
                for item in history
            ) / total_attempts,
            2
        )
    else:
        latest_score = "No attempts"
        average_percentage = 0

    return render_template(
        "quiz_history.html",
        user_name=session["user"],
        history=history,
        total_attempts=total_attempts,
        latest_score=latest_score,
        average_percentage=average_percentage
    )

    
# ============================================================
# WEAK TOPIC ANALYSIS PAGE
# ============================================================


# ============================================================
# WEAK TOPIC ANALYSIS WITH RESOURCE RECOMMENDATIONS
# ============================================================

@app.route("/weak-topics")
def weak_topics():
    if "user" not in session:
        return redirect(url_for("login"))

    question_history_file = os.path.join(
        BASE_DIR, "data", "quiz_question_history.csv"
    )

    resources_file = os.path.join(
        BASE_DIR, "data", "resources.csv"
    )

    topics = []
    topics_analyzed = 0
    weak_topics_count = 0
    overall_accuracy = 0
    has_data = False

    # Match quiz topic names to subjects in resources.csv.
    topic_subject_map = {
        "Python Programming": "Python",
        "Web Development": "Web Development",
        "Data Analysis": "Data Analytics",
        "Data Structures": "Python",
        "Computer Fundamentals": None
    }

    # Read only the logged-in student's quiz answers.
    if os.path.exists(question_history_file):
        try:
            df = pd.read_csv(question_history_file)

            required_columns = {
                "student", "topic", "is_correct"
            }

            if required_columns.issubset(df.columns):
                df = df[
                    df["student"].astype(str).str.strip().str.lower()
                    == session["user"].strip().lower()
                ].copy()

                if not df.empty:
                    df["is_correct"] = (
                        df["is_correct"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["true", "1", "yes"])
                    )

                    df = df.dropna(subset=["topic"])
                    df = df[
                        df["topic"].astype(str).str.strip() != ""
                    ]

                    if not df.empty:
                        has_data = True

                        overall_accuracy = round(
                            df["is_correct"].sum() / len(df) * 100,
                            2
                        )

                        for topic_name, group in df.groupby("topic"):
                            total = len(group)
                            correct = int(group["is_correct"].sum())
                            accuracy = round(correct / total * 100, 2)

                            topics.append({
                                "name": str(topic_name),
                                "total": total,
                                "correct": correct,
                                "accuracy": accuracy,
                                "needs_practice": accuracy < 60,
                                "resources": [],
                                "resource_note": ""
                            })

                        topics.sort(
                            key=lambda item: item["accuracy"]
                        )

                        topics_analyzed = len(topics)

                        weak_topics_count = sum(
                            1 for topic in topics
                            if topic["needs_practice"]
                        )

        except (OSError, pd.errors.EmptyDataError, ValueError) as e:
            print("Weak topic analysis error:", e)

    # Load existing learning resources without modifying the CSV.
    if topics and os.path.exists(resources_file):
        try:
            resources_df = pd.read_csv(resources_file)

            required_resource_columns = {
                "resource_id",
                "title",
                "subject",
                "level",
                "type",
                "difficulty",
                "rating"
            }

            if required_resource_columns.issubset(
                resources_df.columns
            ):
                resources_df = resources_df.fillna("")

                for topic in topics:
                    if not topic["needs_practice"]:
                        continue

                    subject = topic_subject_map.get(topic["name"])

                    if not subject:
                        topic["resource_note"] = (
                            "There is no directly matching resource "
                            "for this topic in your current resource "
                            "dataset yet."
                        )
                        continue

                    matches = resources_df[
                        resources_df["subject"].astype(str).str.strip().str.lower()
                        == subject.lower()
                    ].copy()

                    # Prefer beginner-friendly resources first.
                    level_order = {
                        "beginner": 0,
                        "intermediate": 1,
                        "advanced": 2
                    }

                    matches["_level_order"] = (
                        matches["level"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map(level_order)
                        .fillna(3)
                    )

                    matches["_rating_order"] = pd.to_numeric(
                        matches["rating"], errors="coerce"
                    ).fillna(0)

                    matches = matches.sort_values(
                        by=["_level_order", "_rating_order"],
                        ascending=[True, False]
                    ).head(3)

                    topic["resources"] = matches[
                        [
                            "resource_id",
                            "title",
                            "subject",
                            "level",
                            "type",
                            "difficulty",
                            "rating"
                        ]
                    ].to_dict(orient="records")

                    if topic["name"] == "Data Structures":
                        topic["resource_note"] = (
                            "Your current dataset has no dedicated "
                            "Data Structures resource. These Python "
                            "resources are foundational study material "
                            "while you add a specific stacks and queues "
                            "resource."
                        )
                    elif topic["resources"]:
                        topic["resource_note"] = (
                            "Selected from your existing resources, "
                            "with beginner-friendly options prioritized."
                        )
                    else:
                        topic["resource_note"] = (
                            "No resources matched this subject in "
                            "your current dataset."
                        )

        except (OSError, pd.errors.EmptyDataError, ValueError) as e:
            print("Resource recommendation error:", e)

    return render_template(
        "weak_topics.html",
        user_name=session["user"],
        topics=topics,
        topics_analyzed=topics_analyzed,
        weak_topics_count=weak_topics_count,
        overall_accuracy=overall_accuracy,
        has_data=has_data
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
