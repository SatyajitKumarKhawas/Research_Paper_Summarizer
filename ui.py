import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load API key
load_dotenv()

# Initialize model
model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

st.set_page_config(page_title="📚 Research Paper Explainer", layout="centered")
st.title("🧠 AI-Powered Research Paper Explainer")

st.markdown("Use this tool to generate a customized, understandable summary of landmark AI research papers.")

# --- INPUT SECTION ---
st.subheader("🎯 Select Your Preferences")

col1, col2 = st.columns(2)

with col1:
    paper_input = st.selectbox("📄 Research Paper", [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
        "Vision Transformers (ViT)",
        "AlphaFold: Solving the Protein Folding Problem",
        "ResNet: Deep Residual Learning for Image Recognition",
        "CLIP: Connecting Text and Images",
        "DALL·E: Creating Images from Text",
        "Segment Anything Model (SAM)",
        "Reinforcement Learning with Human Feedback (RLHF)",
        "LLaMA: Open and Efficient Foundation Language Models",
        "Chain of Thought Prompting in LLMs"
    ])

    style_input = st.selectbox("🎨 Explanation Style", [
        "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"
    ])

with col2:
    length_input = st.selectbox("📏 Explanation Length", [
        "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"
    ])

    difficulty = st.radio("📚 Difficulty Level", ["Basic", "Intermediate", "Advanced"], index=1)

bullet_points = st.checkbox("🔹 Present in bullet points")
add_applications = st.checkbox("💡 Include real-world applications/examples")

# --- PROMPT TEMPLATE ---
template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input", "bullet_points", "add_applications", "difficulty"],
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}  
Explanation Length: {length_input}  
Difficulty Level: {difficulty}  
Present in bullet points: {bullet_points}  
Include real-world applications/examples: {add_applications}

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using intuitive code snippets where possible.  

2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If any required information is missing, respond with "Insufficient information available."  
Ensure clarity, accuracy, and alignment with the selected tone and preferences.
"""
)

# --- GENERATE PROMPT ---
prompt = template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input,
    bullet_points="Yes" if bullet_points else "No", #extra points
    add_applications="Yes" if add_applications else "No", #extra points
    difficulty=difficulty
)

st.divider()
st.subheader("📬 Generated Prompt")
with st.expander("View Prompt"):
    st.code(prompt, language="markdown")

# --- SUBMIT BUTTON ---
if st.button("🚀 Generate Summary"):
    with st.spinner("Generating summary using LLaMA..."):
        result = model.invoke(prompt)
        st.success("✅ Summary Generated")
        st.markdown("### 📝 Summary")
        st.write(result.content)
