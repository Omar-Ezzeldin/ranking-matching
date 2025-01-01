import os
import fitz  # PyMuPDF
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            text = " ".join([page.get_text() for page in pdf_file])
        return text
    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        return ""

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

def main():
    # Get paths for job description and resume
    job_description_path = input("Enter the path to job description file (PDF or DOCX): ").strip('"')
    
    # Extract job description text
    try:
        job_description_text = extract_text(job_description_path)
    except Exception as e:
        print(f"Error reading job description: {str(e)}")
        return

    # Extract resume texts
    resume_texts = []
    resume_filenames = []

    # Option 1: Process a single resume
    resume_path = input("Enter the path to resume file (PDF or DOCX): ").strip('"')
    if os.path.exists(resume_path):
        try:
            resume_texts.append(extract_text(resume_path))
            resume_filenames.append(os.path.basename(resume_path))
        except Exception as e:
            print(f"Error processing resume {resume_path}: {str(e)}")

    # Option 2: Process multiple resumes from a directory
    resume_dir = input("Enter path to directory containing additional resumes (or press Enter to skip): ").strip('"')
    if resume_dir and os.path.exists(resume_dir):
        for filename in os.listdir(resume_dir):
            if filename.endswith(('.pdf', '.docx')):
                file_path = os.path.join(resume_dir, filename)
                try:
                    resume_texts.append(extract_text(file_path))
                    resume_filenames.append(filename)
                except Exception as e:
                    print(f"Error processing resume {filename}: {str(e)}")

    # Continue with ranking if we have resumes to process
    if resume_texts:
        # Rank resumes
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        match_scores = rank_resumes(job_description_text, resume_texts)
        ranked_resumes = sorted(zip(resume_filenames, match_scores), key=lambda x: x[1], reverse=True)

        # Display results
        print("\nRanking Results:")
        print("Resume\t\t\tMatch Score (%)")
        print("-" * 50)
        for filename, score in ranked_resumes:
            print(f"{filename}\t\t{round(score * 100, 2)}%")
    else:
        print("No resumes found to process!")

if __name__ == "__main__":
    main()
