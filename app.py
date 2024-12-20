import os
import fitz  # PyMuPDF
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as pdf_file:
        text = " ".join([page.get_text() for page in pdf_file])
    return text

# Function to extract text from Word documents
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from files
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

# Function to rank resumes based on similarity to the job description
def rank_resumes(job_description, resumes):
    # Combine job description and resume texts
    all_texts = [job_description] + resumes

    # Use TF-IDF to vectorize the texts
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Limit features for efficiency
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Extract job description vector and resume vectors
    job_desc_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(job_desc_vector, resume_vectors).flatten()
    return cosine_similarities

# Function to display keyword importance
# def display_keywords(job_description, vectorizer):
#     tfidf_matrix = vectorizer.fit_transform([job_description])
#     feature_array = vectorizer.get_feature_names_out()
#     tfidf_scores = tfidf_matrix.toarray().flatten()
#     top_indices = tfidf_scores.argsort()[-10:][::-1]  # Get top 10 keywords
#     return [(feature_array[i], tfidf_scores[i]) for i in top_indices]

# Upload job description and resumes
print("Upload Job Description (PDF or DOCX):")
job_description_file = files.upload()

print("\nUpload Resumes (PDF or DOCX):")
resume_files = files.upload()

# Extract job description text
job_description_text = ""
for filename, content in job_description_file.items():
    with open(filename, 'wb') as f:
        f.write(content)
    job_description_text = extract_text(filename)
    os.remove(filename)  # Clean up

# Extract resume texts
resume_texts = []
resume_filenames = []
for filename, content in resume_files.items():
    with open(filename, 'wb') as f:
        f.write(content)
    resume_texts.append(extract_text(filename))
    resume_filenames.append(filename)
    os.remove(filename)  # Clean up

# Rank resumes
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
match_scores = rank_resumes(job_description_text, resume_texts)
ranked_resumes = sorted(zip(resume_filenames, match_scores), key=lambda x: x[1], reverse=True)

# Display results
print("\nRanking Results:")
print("Resume\t\t\tMatch Score (%)")
for filename, score in ranked_resumes:
    print(f"{filename}\t\t{round(score * 100, 2)}")

# # Display important job-specific keywords
# print("\nImportant Job-Specific Keywords:")
# top_keywords = display_keywords(job_description_text, vectorizer)
# for keyword, score in top_keywords:
#     print(f"{keyword}: {round(score, 4)}")
