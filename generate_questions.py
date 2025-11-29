import pandas as pd
import pickle

def generate_questions(input_csv="train.csv"):
    df = pd.read_csv(input_csv)

    # Combine question1 + question2
    questions = pd.concat([df['question1'], df['question2']], ignore_index=True)

    # Remove duplicates & NaNs
    questions = questions.dropna().drop_duplicates().reset_index(drop=True)

    # Save all questions
    with open("questions.pkl", "wb") as f:
        pickle.dump(questions.tolist(), f)

    print(f"Saved {len(questions)} unique questions → questions.pkl")

if __name__ == "__main__":
    generate_questions()
