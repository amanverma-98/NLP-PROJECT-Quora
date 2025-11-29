import pickle
import scipy.sparse as sp

def generate_vectors(cv_path="cv.pkl", questions_path="questions.pkl"):
    # Load vectorizer
    with open(cv_path, "rb") as f:
        cv = pickle.load(f)

    # Load questions
    with open(questions_path, "rb") as f:
        questions = pickle.load(f)

    # Vectorize all questions
    vectors = cv.transform(questions)

    # Save vectors
    with open("question_vectors.pkl", "wb") as f:
        pickle.dump(vectors, f)

    print("Saved vectorized questions → question_vectors.pkl")
    print(f"Shape: {vectors.shape}")

if __name__ == "__main__":
    generate_vectors()
