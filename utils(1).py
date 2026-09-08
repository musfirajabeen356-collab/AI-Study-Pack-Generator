"""Utility functions for the AI Study Pack Generator."""

from pathlib import Path
from typing import Optional, Tuple


def extract_text_from_file(uploaded_file) -> Tuple[str, Optional[str]]:
    """Extract text from TXT or PDF uploads.

    Returns:
        (text, error_message)
    """
    if uploaded_file is None:
        return "", None

    try:
        path = Path(uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore"), None

        if suffix == ".pdf":
            import PyPDF2

            with path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n".join(pages).strip()
            if not text:
                return "", "The PDF contains no extractable text."
            return text, None

        return "", "Unsupported file type. Please upload PDF or TXT."

    except Exception as exc:
        return "", f"Could not read the file: {exc}"


def format_study_pack(pack: dict) -> str:
    """Convert the final JSON study pack into readable Markdown."""

    lines = [
        f"# {pack.get('title', 'AI Study Pack')}",
        "",
        "## Summary",
        pack.get("summary", ""),
        "",
        "## Learning Objectives",
    ]

    for item in pack.get("learning_objectives", []):
        lines.append(f"- {item}")

    lines += ["", "## Quick Notes"]

    notes = pack.get("notes", [])
    if isinstance(notes, dict):
        notes = [notes]

    for note in notes:
        if isinstance(note, dict):
            lines += [
                f"### {note.get('concept', 'Concept')}",
                note.get("explanation", ""),
            ]
            if note.get("example"):
                lines.append(f"**Example:** {note['example']}")
        else:
            lines.append(f"- {note}")

    lines += ["", "## Key Terms"]
    for item in pack.get("key_terms", []):
        if isinstance(item, dict):
            lines.append(f"- **{item.get('term', '')}:** {item.get('definition', '')}")
        else:
            lines.append(f"- {item}")

    lines += ["", "## Flashcards"]
    for i, card in enumerate(pack.get("flashcards", []), 1):
        lines += [
            f"**{i}. Q:** {card.get('question', '')}",
            f"**A:** {card.get('answer', '')}",
            "",
        ]

    lines += ["## Practice MCQs"]
    for i, q in enumerate(pack.get("mcqs", []), 1):
        lines.append(f"**{i}. {q.get('question', '')}**")
        options = q.get("options", {})
        if isinstance(options, dict):
            for key, value in options.items():
                lines.append(f"- **{key}.** {value}")
        elif isinstance(options, list):
            for value in options:
                lines.append(f"- {value}")
        lines += [
            f"**Correct answer:** {q.get('correct_answer', '')}",
            f"**Explanation:** {q.get('explanation', '')}",
            "",
        ]

    lines += ["## Short Answer Questions"]
    for i, q in enumerate(pack.get("short_answers", []), 1):
        lines += [
            f"**{i}. {q.get('question', '')}**",
            f"Model answer: {q.get('model_answer', '')}",
            "",
        ]

    lines += ["## Study Plan"]
    plan = pack.get("study_plan", [])
    if isinstance(plan, list):
        for item in plan:
            if isinstance(item, dict):
                lines.append(
                    f"- **{item.get('day', item.get('session', 'Session'))}:** "
                    f"{item.get('focus', item.get('activity', ''))}"
                )
            else:
                lines.append(f"- {item}")

    lines += ["", "## Final Review Points"]
    for item in pack.get("final_review_points", []):
        lines.append(f"- {item}")

    lines += [
        "",
        f"**Quality score:** {pack.get('quality_score', 'N/A')}/100"
    ]

    return "\n".join(lines)


def safe_int(value, default=10, minimum=5, maximum=20):
    try:
        number = int(value)
        return max(minimum, min(maximum, number))
    except (TypeError, ValueError):
        return default
