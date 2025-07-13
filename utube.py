import os
import getpass
import re
from typing import TypedDict
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from langdetect import detect

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph

# STEP 1: Get Groq API key
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# STEP 2: Load LLaMA3 model
llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.environ["GROQ_API_KEY"]
)

# STEP 3: LangGraph State
class MyState(TypedDict):
    video_url: str
    video_text: str
    detected_language: str
    summary: str

# STEP 4: Extract + clean transcript
def extract_transcript(state: MyState) -> MyState:
    url = state["video_url"]
    video_id = parse_qs(urlparse(url).query).get("v", [None])[0]

    if not video_id:
        raise ValueError("❌ Invalid YouTube URL")

    transcript = None

    try:
        # Try direct English transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        print("✅ English transcript found.")
    except NoTranscriptFound:
        print("⚠️ English transcript not found. Trying fallback options...")

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            fallback = transcript_list.find_transcript(['hi', 'en'])

            # fetch returns a list of segments
            transcript = fallback.fetch()
        except Exception as e:
            raise RuntimeError(f"❌ Failed to fetch any transcript: {str(e)}")

    # Ensure it's a list of dicts
    if not isinstance(transcript, list) or not all(isinstance(t, dict) for t in transcript):
        raise TypeError("❌ Transcript is not in expected format (list of dicts).")

    # Combine transcript
    full_text = " ".join([t["text"] for t in transcript])

    # Clean: remove [Music], [Applause], etc.
    cleaned_text = re.sub(r"\[(.*?)\]", "", full_text).strip()

    # Detect language
    detected_lang = detect(cleaned_text)
    print(f"🌍 Detected Language: {detected_lang}")
    print("\n📄 Transcript Extracted (first 500 chars):\n", cleaned_text[:500], "...\n")

    return {
        "video_text": cleaned_text,
        "detected_language": detected_lang,
        **state
    }

# STEP 5: Translate to English (if needed)
def translate_to_english(state: MyState) -> MyState:
    if state["detected_language"] == "en":
        print("✅ Already in English. No translation needed.")
        return state

    prompt = f"Translate this text to English:\n\n{state['video_text']}"
    result = llm.invoke(prompt)
    print("🌐 Translated to English.")

    return {
        "video_text": result.content,
        **state
    }

# STEP 6: Summarize transcript
def summarize_transcript(state: MyState) -> MyState:
    prompt = f"Summarize this YouTube video transcript into note form and give a whole short type summary :\n\n{state['video_text']}"
    result = llm.invoke(prompt)

    print("\n📝 Summary:\n", result.content)

    return {
        "summary": result.content,
        **state
    }

# STEP 7: Build LangGraph
builder = StateGraph(state_schema=MyState)

builder.add_node("extract", extract_transcript)
builder.add_node("translate", translate_to_english)
builder.add_node("summarize", summarize_transcript)

builder.set_entry_point("extract")
builder.add_edge("extract", "translate")
builder.add_edge("translate", "summarize")
builder.set_finish_point("summarize")

graph = builder.compile()

# STEP 8: Run It
if __name__ == "__main__":
    # ✅ Replace with any YouTube URL
    video_url = "https://www.youtube.com/watch?v=C8IfGDrmwiE"

    print("🚀 Running LangGraph...")
    try:
        final_state = graph.invoke({"video_url": video_url})

        print("\n✅ Final State:")
        print("📺 Video URL:", final_state["video_url"])
        print("🌍 Language:", final_state["detected_language"])
        print("📄 Transcript (preview):", final_state["video_text"][:300])
        print("\n📝 Summary:\n", final_state["summary"])

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
