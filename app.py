import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re
from dotenv import load_dotenv
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube AI Summarizer",
    page_icon="🎬",
    layout="wide"
)

# ── API Key ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Please add it in Space Settings → Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str) -> tuple[str, list]:
    """Fetch transcript and return full text + timestamped chunks."""
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id)
    transcript_list = [{"text": s.text, "start": s.start} for s in fetched]
    full_text = " ".join([t["text"] for t in transcript_list])
    return full_text, transcript_list

def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def summarize(transcript: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""You are an expert content summarizer.
Analyze the following YouTube video transcript and provide:

1. **📋 Overview** (2-3 sentences about what the video is about)

2. **🔑 Key Points** (5-7 most important points as bullet points)

3. **💡 Main Takeaways** (3 actionable insights the viewer should remember)

4. **🏷️ Topics Covered** (list the main topics as tags)

Transcript:
{transcript}

Provide a clear, well-structured summary:"""
    )
    chain = prompt | llm
    return chain.invoke({"transcript": transcript[:12000]}).content

def generate_timestamps(transcript_list: list) -> str:
    """Generate key timestamps using GPT."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   
    # Sample transcript with timestamps (every 30 seconds)
    sampled = []
    last_time = -30
    for t in transcript_list:
        if t["start"] - last_time >= 30:
            sampled.append(f"[{format_time(t['start'])}] {t['text']}")
            last_time = t["start"]
   
    sampled_text = "\n".join(sampled[:80])
   
    prompt = f"""Based on these transcript excerpts with timestamps, identify 6-8 key moments in the video.
Format each as: MM:SS - Brief description of what's being discussed

Transcript excerpts:
{sampled_text}

Key timestamps:"""
   
    return llm.invoke(prompt).content

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎬 YouTube AI Video Summarizer")
st.caption("Paste any YouTube URL and get an instant AI-powered summary, key points, and timestamps.")

# URL Input
url = st.text_input(
    "🔗 YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    summarize_btn = st.button("✨ Summarize Video", type="primary", use_container_width=True)
with col2:
    timestamps_btn = st.button("⏱️ Generate Timestamps", use_container_width=True)
with col3:
    both_btn = st.button("🚀 Full Analysis", use_container_width=True)

st.divider()

# ── Processing ────────────────────────────────────────────────────────────────
if summarize_btn or timestamps_btn or both_btn:
    if not url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
        else:
            # Show video thumbnail
            st.image(
                 f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                 width=700

            )
            st.markdown(f"🔗 [Watch on YouTube]({url})")
            st.divider()

            # Fetch transcript
            with st.spinner("📥 Fetching video transcript..."):
                try:
                    transcript_text, transcript_list = get_transcript(video_id)
                    word_count = len(transcript_text.split())
                    duration = format_time(transcript_list[-1]["start"]) if transcript_list else "N/A"

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("📝 Words in Transcript", f"{word_count:,}")
                    with col_b:
                        st.metric("⏱️ Video Duration", f"~{duration}")

                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                    st.stop()
                except NoTranscriptFound:
                    st.error("❌ No transcript found for this video.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Error fetching transcript: {str(e)}")
                    st.stop()

            # Summary
            if summarize_btn or both_btn:
                with st.spinner("🤖 Generating AI summary..."):
                    summary = summarize(transcript_text)
                st.markdown("## 📋 AI Summary")
                st.markdown(summary)
                st.divider()

            # Timestamps
            if timestamps_btn or both_btn:
                with st.spinner("⏱️ Generating timestamps..."):
                    timestamps = generate_timestamps(transcript_list)
                st.markdown("## ⏱️ Key Timestamps")
                st.markdown(timestamps)
                st.divider()

            # Raw transcript expander
            with st.expander("📄 View Raw Transcript"):
                st.text_area("Transcript", transcript_text, height=300)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using LangChain + OpenAI + Streamlit | by Sakshi Mishra")