# ============================================================
# SMART EDU - DYNAMIC AI ANSWER ENGINE
# ============================================================

import os
import json
from google import genai


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# DEFAULT RESPONSE
# ============================================================

def make_error(message):
    return {
        "type": "error",
        "question_type": "General",
        "answer": message,
        "explanation": "",
        "code": "",
        "example": "",
        "key_takeaway": ""
    }


# ============================================================
# GENERATE ANSWER FOR ANY EDUCATIONAL QUESTION
# ============================================================

def generate_answer(text, topic="General"):

    if not text or not text.strip():
        return make_error(
            "No readable question was detected. "
            "Please upload a clearer image."
        )

    if client is None:
        return make_error(
            "Gemini API key is missing. "
            "Set GEMINI_API_KEY and restart Smart Edu."
        )

    prompt = f"""
You are Smart Edu, an educational AI assistant.

Answer the actual question found in the OCR text below.
The question can belong to ANY academic subject.

Supported examples include:
- Programming and code output
- Mathematics and numerical problems
- Theory and definitions
- Multiple-choice questions
- Science and engineering
- Databases, networking, and operating systems
- AI and machine learning
- Other educational subjects

Detected topic: {topic}

OCR text:
{text}

Instructions:
1. Identify and answer the actual question.
2. Do not assume that every question is about Python.
3. For code-output questions, trace the supplied code carefully.
4. For numerical questions, show the necessary calculations.
5. For MCQs, state the correct option when the options are readable.
6. For theory questions, explain the concept clearly.
7. Preserve the meaning of supplied code, equations, and values.
8. Do not invent missing information or silently correct unclear text.
9. If the question is incomplete or ambiguous, state what is unclear.
10. Use explanations appropriate for a university student.
11. Include solution code only when it is useful.
12. Return valid JSON only, without Markdown code fences.

Return exactly these fields:
{{
    "question_type": "Programming",
    "answer": "Direct answer",
    "explanation": "Explanation or solution steps",
    "code": "",
    "example": "",
    "key_takeaway": "Main learning point"
}}

question_type must be one of:
Programming, Numerical, Theory, MCQ, General.

Use empty strings for fields that are not applicable.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        raw_response = (response.text or "").strip()

        if not raw_response:
            return make_error(
                "The AI returned an empty response. Please try again."
            )

        # Remove accidental Markdown JSON fences.
        if raw_response.startswith("```"):
            raw_response = raw_response.replace(
                "```json", "", 1
            )
            raw_response = raw_response.replace(
                "```", ""
            ).strip()

        data = json.loads(raw_response)

        valid_types = {
            "Programming", "Numerical", "Theory", "MCQ", "General"
        }

        question_type = data.get("question_type", "General")

        if question_type not in valid_types:
            question_type = "General"

        return {
            "type": "ai",
            "question_type": question_type,
            "answer": str(data.get("answer", "")),
            "explanation": str(data.get("explanation", "")),
            "code": str(data.get("code", "")),
            "example": str(data.get("example", "")),
            "key_takeaway": str(data.get("key_takeaway", ""))
        }

    except Exception as e:
        print("GEMINI API ERROR:", type(e).__name__, str(e))

        return make_error(
            "Unable to generate an answer right now. "
            "Check your Gemini API configuration and try again."
        )

        
# ============================================================
# AI FLASHCARD GENERATOR - SMART EDU
# ============================================================

def generate_flashcards(text, topic="General", count=10):

    if not text or not text.strip():
        return {
            "success": False,
            "message": "Please provide readable study material.",
            "flashcards": []
        }

    if client is None:
        return {
            "success": False,
            "message": (
                "Gemini API key is missing. "
                "Configure GEMINI_API_KEY and restart Smart Edu."
            ),
            "flashcards": []
        }

    # Keep the requested number within a reasonable range.
    count = max(1, min(int(count), 20))

    prompt = f"""
You are the Smart Edu educational flashcard generator.

Create exactly {count} useful flashcards from the supplied
study material.

Topic: {topic}

Study material:
{text}

Rules:
1. Use only facts supported by the supplied material.
2. Each question must test one clear concept.
3. Answers must be concise, correct, and useful for revision.
4. Avoid duplicate questions.
5. Do not create questions about information absent from the material.
6. If the material is too short, create fewer cards rather than
   inventing facts.
7. Return valid JSON only, without Markdown code fences.

Use this JSON structure:
{{
    "flashcards": [
        {{
            "question": "What is ...?",
            "answer": "..."
        }}
    ]
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        raw_response = (response.text or "").strip()

        if not raw_response:
            raise ValueError("The AI returned an empty response.")

        # Remove accidental Markdown JSON fences.
        if raw_response.startswith("```"):
            raw_response = raw_response.replace(
                "```json", "", 1
            )
            raw_response = raw_response.replace(
                "```", ""
            ).strip()

        data = json.loads(raw_response)
        cards = data.get("flashcards", [])

        if not isinstance(cards, list):
            raise ValueError("Invalid flashcard response format.")

        valid_cards = []

        for card in cards:
            if not isinstance(card, dict):
                continue

            question = str(card.get("question", "")).strip()
            answer = str(card.get("answer", "")).strip()

            if question and answer:
                valid_cards.append({
                    "question": question,
                    "answer": answer
                })

        if not valid_cards:
            raise ValueError("No valid flashcards were generated.")

        return {
            "success": True,
            "message": "Flashcards generated successfully.",
            "topic": topic,
            "flashcards": valid_cards[:count]
        }

    except Exception as e:
        print(
            "FLASHCARD GENERATION ERROR:",
            type(e).__name__,
            str(e)
        )

        return {
            "success": False,
            "message": (
                "Unable to generate flashcards right now. "
                "Check your Gemini API configuration and try again."
            ),
            "flashcards": []
        }
