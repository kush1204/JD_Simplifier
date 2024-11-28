import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import chardet
import spacy
from spacy.matcher import PhraseMatcher
from pathlib import Path
import pandas as pd
import subprocess




# Set up cache directories for models
model_cache_dir = Path("./models")
model_cache_dir.mkdir(parents=True, exist_ok=True)

# Load summarization model (T5-small)
summary_model_name = "t5-small"
summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name, cache_dir=model_cache_dir)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name, cache_dir=model_cache_dir)

# Load similarity model (sentence-transformers)
similarity_model_name = "sentence-transformers/all-MiniLM-L6-v2"
similarity_tokenizer = AutoTokenizer.from_pretrained(similarity_model_name, cache_dir=model_cache_dir)
similarity_model = AutoModel.from_pretrained(similarity_model_name, cache_dir=model_cache_dir)

# Load spaCy NER model

# def load_spacy_model():
#     try:
#         nlp = spacy.load("./models/en_core_web_sm")
#     except OSError:
#         # Install the model if it's not already installed
#         subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
#         nlp = spacy.load("en_core_web_sm")
#     return nlp

# # Load spaCy NER model from local directory
# def load_spacy_model():
#     model_path = Path("./models/en_core_web_sm")

#     # Check if the model exists in the specified directory
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found in {model_path}. Please upload the model directory.")
    
#     # Load the model from the local directory
#     nlp = spacy.load(model_path)
#     return nlp


# nlp = load_spacy_model()




# Pre-defined lists for NER matching
SKILL_KEYWORDS = ["skill", "qualification", "requirement", "proficiency", "knowledge", "expertise"]
EXPERIENCE_KEYWORDS = ["experience", "years", "background", "track record"]
RESPONSIBILITY_KEYWORDS = ["responsibility", "duty", "task", "accountability", "role"]

# Utility to extract text from uploaded files
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages)
    elif file.type == "text/plain":
        content = file.read()
        detected_encoding = chardet.detect(content)
        text = content.decode(detected_encoding['encoding'])
    else:
        st.error("Unsupported file type. Please upload a PDF or text file.")
        return None
    return text

# Extract details using spaCy and phrase matcher
# def extract_with_ner(jd_text):
#     doc = nlp(jd_text)
#     matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

#     skill_patterns = [nlp.make_doc(text) for text in SKILL_KEYWORDS]
#     experience_patterns = [nlp.make_doc(text) for text in EXPERIENCE_KEYWORDS]
#     responsibility_patterns = [nlp.make_doc(text) for text in RESPONSIBILITY_KEYWORDS]

#     matcher.add("SKILLS", None, *skill_patterns)
#     matcher.add("EXPERIENCE", None, *experience_patterns)
#     matcher.add("RESPONSIBILITIES", None, *responsibility_patterns)

#     skills, experience, responsibilities = [], [], []

#     for sent in doc.sents:
#         matches = matcher(sent)
#         for match_id, start, end in matches:
#             match_label = nlp.vocab.strings[match_id]
#             if match_label == "SKILLS":
#                 skills.append(sent.text.strip())
#             elif match_label == "EXPERIENCE":
#                 experience.append(sent.text.strip())
#             elif match_label == "RESPONSIBILITIES":
#                 responsibilities.append(sent.text.strip())

#     skills = list(dict.fromkeys(skills))
#     experience = [item for item in dict.fromkeys(experience) if item not in skills]
#     responsibilities = [
#         item for item in dict.fromkeys(responsibilities)
#         if item not in skills and item not in experience
#     ]

#     return {
#         "skills": list(dict.fromkeys(skills)),
#         "experience": list(dict.fromkeys(experience)),
#         "responsibilities": list(dict.fromkeys(responsibilities)),
#     }

# Apply custom CSS for white font color
st.markdown(
    """
    <style>
    .main {
        background-color: #262730;
        color: white;
    }
    div.stButton > button {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Job Description Summarizer and Comparator")

# Sidebar for inputs
st.sidebar.title("JD Input")
# Main features
st.sidebar.title("Select a Feature")
feature = st.sidebar.radio("Choose", ["Summarize JD", "Extract Details", "Compare JD & Resume"])

jd_file = st.sidebar.file_uploader("Upload Job Description (PDF or Text)", type=["pdf", "txt"])
jd_text_input = st.sidebar.text_area("Or Paste Job Description Text")

# Check if file or text is provided
if jd_file:
    jd_text = extract_text(jd_file)
elif jd_text_input:
    jd_text = jd_text_input.strip()
else:
    jd_text = None
    st.sidebar.warning("Please upload a file or paste the job description text.")

resume_file = st.sidebar.file_uploader("Upload Resume (PDF or Text)", type=["pdf", "txt"])



if jd_text:
    if feature == "Summarize JD":
        st.header("Summarize Job Description")
        inputs = summary_tokenizer("summarize: " + jd_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = summary_model.generate(inputs.input_ids, max_length=200, min_length=40, length_penalty=1.0, num_beams=4, early_stopping=True)
        summary = summary_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)

    elif feature == "Compare JD & Resume" and resume_file:
        st.header("Compare Job Description and Resume")
        resume_text = extract_text(resume_file)
        if resume_text:
            jd_embedding = similarity_model(**similarity_tokenizer(jd_text, return_tensors="pt", truncation=True, padding=True)).last_hidden_state.mean(dim=1)
            resume_embedding = similarity_model(**similarity_tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)).last_hidden_state.mean(dim=1)
            similarity = cosine_similarity(jd_embedding.detach().numpy(), resume_embedding.detach().numpy())[0][0]
            st.subheader("Similarity Score:")
            st.write(f"{round(similarity * 100, 2)}%")
            st.write("Good alignment!" if similarity > 0.7 else "Consider tailoring your resume further.")

    # elif feature == "Extract Details":
    #     st.header("Extract Details from JD")
    #     details = extract_with_ner(jd_text)
    #     st.subheader("Extracted Details:")
    #     st.json(details)

    # import pandas as pd

    # elif feature == "Extract Details":
    # # Title
    #     st.header("Extract Details from JD")

    #     # Ensure JD text is available
    #     if jd_text.strip():
    #         # Extract details using NER
    #         details = extract_with_ner(jd_text)

    #         # Prepare DataFrame for display
    #         data = {
    #             "Category": [],
    #             "Text": []
    #         }
    #         for skill in details["skills"]:
    #             data["Category"].append("Skill")
    #             data["Text"].append(skill)
    #         for experience in details["experience"]:
    #             data["Category"].append("Experience")
    #             data["Text"].append(experience)
    #         for responsibility in details["responsibilities"]:
    #             data["Category"].append("Responsibility")
    #             data["Text"].append(responsibility)

    #         # Convert data to a DataFrame
    #         df = pd.DataFrame(data)

    #         # Add dropdown for category selection
    #         st.subheader("Select Category")
    #         selected_category = st.selectbox(
    #             "Filter by category:",
    #             options=["All", "Skill", "Experience", "Responsibility"]
    #         )

    #         # Filter DataFrame based on selection
    #         if selected_category != "All":
    #             filtered_df = df[df["Category"] == selected_category]
    #         else:
    #             filtered_df = df

    #         # Display the filtered data
    #         st.write("### Extracted Information")
    #         st.table(filtered_df)

    #         # Provide a download option for the data
    #         st.download_button(
    #             label="Download Extracted Details as CSV",
    #             data=filtered_df.to_csv(index=False),
    #             file_name="extracted_details.csv",
    #             mime="text/csv"
    #         )
    #     else:
    #         st.error("No Job Description text provided.")




else:
    st.warning("Please upload a file or paste JD text to proceed.")
