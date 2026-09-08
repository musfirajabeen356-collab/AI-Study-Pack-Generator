import os
import streamlit as st

from workflow import StudyPackWorkflow, WorkflowContext
from utils import extract_text_from_file, format_study_pack, safe_int


st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Study Pack Generator")
st.caption(
    "Gemini-powered multi-stage workflow: Planning → Content → Assessment → Review → Refinement"
)

with st.sidebar:
    st.header("⚙️ Study Settings")

    topic = st.text_input(
        "Study topic",
        placeholder="e.g. Photosynthesis, Python OOP, World War II",
    )

    level = st.selectbox(
        "Student level",
        ["School", "College/University", "Beginner", "Intermediate", "Advanced"],
    )

    difficulty = st.select_slider(
        "Difficulty",
        options=["Easy", "Medium", "Hard"],
        value="Medium",
    )

    language = st.selectbox(
        "Language",
        ["English", "Urdu", "Roman Urdu"],
    )

    learning_goal = st.text_area(
        "Learning goal",
        placeholder="e.g. Prepare for an exam and understand the concepts",
    )

    time_available = st.selectbox(
        "Available study time",
        ["1 day", "3 days", "1 week", "2 weeks"],
        index=1,
    )

    question_count = st.slider(
        "Flashcards / MCQs",
        min_value=5,
        max_value=20,
        value=10,
    )

    source_file = st.file_uploader(
        "Optional study material",
        type=["pdf", "txt"],
    )

    generate = st.button(
        "🚀 Generate Study Pack",
        type="primary",
        use_container_width=True,
    )

if generate:
    if not topic.strip():
        st.error("Please enter a study topic.")
        st.stop()

    source_text, file_error = extract_text_from_file(source_file)

    if file_error:
        st.warning(file_error)

    if len(source_text) > 20000:
        source_text = source_text[:20000]
        st.warning("Uploaded material was trimmed to 20,000 characters.")

    # Streamlit Secrets first, environment variable second.
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY is missing.")
        st.code('GEMINI_API_KEY = "your-gemini-api-key"', language="toml")
        st.stop()

    context = WorkflowContext(
        topic=topic.strip(),
        level=level,
        language=language,
        difficulty=difficulty,
        question_count=safe_int(question_count),
        source_text=source_text,
        learning_goal=learning_goal.strip(),
        time_available=time_available,
    )

    workflow = StudyPackWorkflow(api_key=api_key)

    stages = [
        ("Planning", workflow.planning_stage),
        ("Content Generation", workflow.content_generation_stage),
        ("Assessment", workflow.assessment_stage),
        ("Review", workflow.review_stage),
        ("Refinement", workflow.refinement_stage),
    ]

    st.subheader("🔄 AI Workflow Progress")
    progress = st.progress(0)
    status = st.empty()

    failed = False

    for index, (name, stage_function) in enumerate(stages, start=1):
        status.info(f"Running **Stage {index}: {name}**...")

        try:
            context = stage_function(context)
            progress.progress(index / len(stages))

            if name == "Review":
                score = context.review.get("score", "N/A")
                approved = context.review.get("approved", False)
                st.success(
                    f"✓ Review completed — Quality: {score}/100 | Approved: {approved}"
                )
            else:
                st.success(f"✓ {name} completed.")

        except Exception as exc:
            failed = True
            message = f"{name} stage failed: {exc}"
            context.errors.append(message)
            status.error(message)
            break

    if failed:
        st.error(
            "The workflow stopped safely. Check your Gemini API key, "
            "model availability, internet connection, or uploaded material."
        )
        with st.expander("Error details"):
            for error in context.errors:
                st.write(f"- {error}")
        st.stop()

    status.success("🎉 All 5 AI stages completed successfully!")

    tab_final, tab_plan, tab_review, tab_context = st.tabs(
        ["📚 Final Study Pack", "🧠 Planning", "🔍 Review", "🧩 Context"]
    )

    with tab_final:
        markdown = format_study_pack(context.refined_pack)
        st.markdown(markdown)

        st.download_button(
            "⬇️ Download Study Pack",
            data=markdown,
            file_name="study_pack.md",
            mime="text/markdown",
        )

    with tab_plan:
        st.json(context.plan)

    with tab_review:
        review = context.review
        st.metric("Quality Score", f"{review.get('score', 'N/A')}/100")

        for key in [
            "factual_issues",
            "ambiguity_issues",
            "level_issues",
            "objective_gaps",
            "duplicate_issues",
            "language_issues",
            "corrections",
        ]:
            values = review.get(key, [])
            if values:
                st.write(f"**{key.replace('_', ' ').title()}**")
                for item in values:
                    st.write(f"- {item}")

    with tab_context:
        st.json({
            "topic": context.topic,
            "level": context.level,
            "difficulty": context.difficulty,
            "language": context.language,
            "question_count": context.question_count,
            "time_available": context.time_available,
            "errors": context.errors,
        })

else:
    st.subheader("How the AI workflow works")

    cols = st.columns(5)

    stages_info = [
        ("1️⃣ Planning", "Creates personalized objectives, prerequisites, concepts and schedule."),
        ("2️⃣ Content", "Generates notes, examples, key terms and flashcards."),
        ("3️⃣ Assessment", "Creates MCQs, short answers and answer keys."),
        ("4️⃣ Review", "Checks accuracy, difficulty, ambiguity, duplicates and coverage."),
        ("5️⃣ Refinement", "Uses review feedback to create the final corrected pack."),
    ]

    for col, (title, description) in zip(cols, stages_info):
        with col:
            st.markdown(f"### {title}")
            st.write(description)

    st.markdown("### 🔐 Gemini API setup")
    st.write("For Streamlit Cloud, add the following under Settings → Secrets:")
    st.code('GEMINI_API_KEY = "your-gemini-api-key"', language="toml")
