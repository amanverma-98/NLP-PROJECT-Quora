import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from helper import preprocess, query_point_creator, cv, test_common_words


# Load Model + DB

@st.cache_resource
def load_model_and_db():
    model = pickle.load(open("model.pkl", "rb"))

    try:
        quora_questions = pickle.load(open("questions.pkl", "rb"))
        quora_vectors = pickle.load(open("question_vectors.pkl", "rb"))
    except:
        quora_questions = []
        quora_vectors = None

    return model, quora_questions, quora_vectors


model, quora_questions, quora_vectors = load_model_and_db()

st.markdown("""
<h1 style='text-align:center;
           font-size: 42px;
           background: -webkit-linear-gradient(#6a11cb, #2575fc);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;'>
Duplicate Question Checker
</h1>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("Features")
page = st.sidebar.radio("Go to", [
    "1️⃣ Check Duplicate",
    "2️⃣ Find Similar Questions",
    "3️⃣ Compare Two Questions"
])



# PAGE 1 - CHECK DUPLICATE EXISTS OR NOT
if page == "1️⃣ Check Duplicate":
    st.title("Check if Question Already Exists on Quora")
    user_q = st.text_area("Enter the question", height=150)

    top_k = st.slider("How many duplicate results you want?", 1, 5, 2)

    if st.button("Check Duplicate"):
        if not user_q.strip():
            st.warning("Please type a question.")
        else:
            cleaned = preprocess(user_q)

            # if we don't have vectors
            if quora_vectors is None:
                st.error("No question_vectors.pkl found. Upload first.")
            else:
                # BOW or Word2Vec (depending on your cv)
                user_vec = cv.transform([cleaned])

                # cosine similarity
                sims = cosine_similarity(user_vec, quora_vectors).flatten()

                # top candidates
                idx_sorted = np.argsort(sims)[::-1][:top_k]

                duplicate_found = False
                match_info = None

                # check each candidate with your model
                for idx in idx_sorted:
                    candidate = quora_questions[idx]
                    cand_clean = preprocess(candidate)

                    try:
                        features = query_point_creator(cleaned, cand_clean)
                        pred = model.predict(features)[0]
                        prob = model.predict_proba(features)[0][1]
                    except:
                        pred = 0
                        prob = 0

                    if pred == 1:   # found duplicate
                        duplicate_found = True
                        match_info = (candidate, sims[idx], prob)
                        break

                if duplicate_found:
                    st.success("✔️ This question already exists on Quora!")
                    st.write("### Most likely match:")
                    st.write(f"**{match_info[0]}**")
                    st.write(f"Similarity: `{match_info[1]:.3f}`")
                    st.write(f"Duplicate probability: `{match_info[2]:.3f}`")
                else:
                    st.info("🆕 This seems to be a NEW question.")




# PAGE 2 - LIST SIMILAR QUESTIONS

elif page == "2️⃣ Find Similar Questions":
    st.title("Find Similar Questions on Quora")

    user_q = st.text_area("Enter your question", height=150)

    top_k = st.slider("Number of similar questions:", 1, 5, 2)

    if st.button("Find Similar"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            cleaned = preprocess(user_q)

            if quora_vectors is None:
                st.error("No question_vectors.pkl file found.")
            else:
                user_vec = cv.transform([cleaned])   # BoW or Word2Vec

                sims = cosine_similarity(user_vec, quora_vectors).flatten()

                idx_sorted = np.argsort(sims)[::-1][:top_k]

                st.write("### Top Similar Questions:")

                for idx in idx_sorted:
                    candidate = quora_questions[idx]
                    cand_clean = preprocess(candidate)

                    try:
                        features = query_point_creator(cleaned, cand_clean)
                        pred = model.predict(features)[0]
                        prob = model.predict_proba(features)[0][1]
                    except:
                        pred = None
                        prob = None

                    with st.container():
                        st.markdown(f"**{candidate}**")
                        st.write(f"Similarity: `{sims[idx]:.3f}`")

                        if pred is not None:
                            if pred == 1:
                                st.success(f"Duplicate (prob={prob:.3f})")
                            else:
                                st.info(f"Not Duplicate (prob={prob:.3f})")

                        st.markdown("---")




# PAGE 3 → DIRECT COMPARE TWO QUESTIONS
elif page == "3️⃣ Compare Two Questions":
    st.title("Compare Two Questions")

    q1 = st.text_area("Question 1", height=120)
    q2 = st.text_area("Question 2", height=120)

    if st.button("Compare"):
        if not q1.strip() or not q2.strip():
            st.warning("Enter both questions.")
        else:
            q1_c = preprocess(q1)
            q2_c = preprocess(q2)

            try:
                features = query_point_creator(q1_c, q2_c)
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]
            except Exception as e:
                st.error(f"Feature error: {e}")
                pred = None

            if pred == 1:
                st.success(f"✔️ Duplicate (prob={prob:.3f})")
            else:
                st.error(f"✖️ Not Duplicate (prob={prob:.3f})")

            st.write("### Extra Info")
            st.write({
                "len_q1": len(q1_c),
                "len_q2": len(q2_c),
                "common_words": test_common_words(q1_c, q2_c)
            })
