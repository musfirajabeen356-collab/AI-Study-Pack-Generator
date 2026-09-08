"""
Multi-stage AI workflow for the personalized study-pack generator.
AI provider: Google Gemini API via the official google-genai SDK.

Stages:
1. Planning
2. Content Generation
3. Assessment
4. Review
5. Refinement

Context is passed from one stage to the next.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os

from google import genai
from google.genai import types


@dataclass
class WorkflowContext:
    topic: str
    level: str
    language: str
    difficulty: str
    question_count: int
    source_text: str = ""
    learning_goal: str = ""
    time_available: str = "3 days"
    plan: Dict[str, Any] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)
    assessment: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=dict)
    refined_pack: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class StudyPackWorkflow:
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to Streamlit Secrets "
                "or set it as an environment variable."
            )

        self.client = genai.Client(api_key=self.api_key)

    def _call_ai(self, system: str, prompt: str) -> Dict[str, Any]:
        """Call Gemini and safely parse a JSON response."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system}\n\n{prompt}",
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

        text = (response.text or "").strip()

        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {text[:500]}"
            ) from exc

    def planning_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        prompt = f"""
Create a personalized learning plan.

Topic: {ctx.topic}
Student level: {ctx.level}
Difficulty: {ctx.difficulty}
Language: {ctx.language}
Learning goal: {ctx.learning_goal or "General understanding and exam preparation"}
Available time: {ctx.time_available}
Source material:
{ctx.source_text[:12000] if ctx.source_text else "None"}

Return JSON with:
- objectives: list of 3-5 measurable learning objectives
- prerequisites: list
- concepts: ordered list from basic to advanced
- study_strategy: short paragraph
- schedule: list of sessions containing session, focus, activity, estimated_minutes
- personalization_notes: list
"""
        ctx.plan = self._call_ai(
            "You are an expert instructional designer. Design realistic, "
            "accurate, age-appropriate personalized learning plans.",
            prompt,
        )
        return ctx

    def content_generation_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        prompt = f"""
Generate study content using the plan below.

PLAN:
{json.dumps(ctx.plan, ensure_ascii=False)}

Topic: {ctx.topic}
Level: {ctx.level}
Difficulty: {ctx.difficulty}
Language: {ctx.language}

SOURCE MATERIAL:
{ctx.source_text[:16000] if ctx.source_text else "No source material provided."}

Return JSON with:
- title
- summary
- notes: list containing concept, explanation, example
- key_terms: list containing term and definition
- flashcards: exactly {ctx.question_count} objects with question and answer
- misconceptions: list containing mistake and correction

If source material is supplied, prioritize it and do not invent source-specific facts.
"""
        ctx.content = self._call_ai(
            "You are an expert educational writer. Produce clear, accurate "
            "study content appropriate to the learner.",
            prompt,
        )
        return ctx

    def assessment_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        prompt = f"""
Create an assessment based on the learning plan and generated content.

LEARNING PLAN:
{json.dumps(ctx.plan, ensure_ascii=False)}

CONTENT:
{json.dumps(ctx.content, ensure_ascii=False)}

Return JSON with:
- mcqs: exactly {ctx.question_count} objects. Each contains:
  question, options (A-D), correct_answer, explanation, difficulty
- short_answers: 5 objects containing question and model_answer
- answer_key: list
- coverage: concepts tested
"""
        ctx.assessment = self._call_ai(
            "You are an assessment specialist. Create valid, unambiguous "
            "questions that test the intended learning objectives.",
            prompt,
        )
        return ctx

    def review_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        prompt = f"""
Audit the study pack for educational quality.

PLAN:
{json.dumps(ctx.plan, ensure_ascii=False)}

CONTENT:
{json.dumps(ctx.content, ensure_ascii=False)}

ASSESSMENT:
{json.dumps(ctx.assessment, ensure_ascii=False)}

Return JSON with:
- score: integer 0-100
- factual_issues: list
- ambiguity_issues: list
- level_issues: list
- objective_gaps: list
- duplicate_issues: list
- language_issues: list
- corrections: list of specific fixes
- approved: boolean

Be strict. Approve only if the material is coherent, useful, appropriately
difficult, and internally consistent.
"""
        ctx.review = self._call_ai(
            "You are a rigorous educational quality reviewer. Find errors "
            "and gaps before material is delivered to a student.",
            prompt,
        )
        return ctx

    def refinement_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        prompt = f"""
Refine the study pack using the review findings.

PLAN:
{json.dumps(ctx.plan, ensure_ascii=False)}

CONTENT:
{json.dumps(ctx.content, ensure_ascii=False)}

ASSESSMENT:
{json.dumps(ctx.assessment, ensure_ascii=False)}

REVIEW:
{json.dumps(ctx.review, ensure_ascii=False)}

Return the FINAL corrected JSON with:
- title
- summary
- learning_objectives
- notes
- key_terms
- flashcards
- mcqs
- short_answers
- study_plan
- final_review_points
- quality_score

Preserve correct material. Fix the issues identified in the review.
Ensure the final flashcard and MCQ counts match the requested count.
"""
        ctx.refined_pack = self._call_ai(
            "You are the final educational editor. Produce a polished, "
            "accurate, learner-ready study pack.",
            prompt,
        )
        return ctx
