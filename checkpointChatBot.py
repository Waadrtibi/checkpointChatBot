import streamlit as st
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# === Prétraitement du texte ===
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)  # Supprimer la ponctuation
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# === Charger les données du CSV ===
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "climate_change_faqs.csv")

    # Lecture avec encodage pour corriger les caractères spéciaux
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # Filtrer uniquement les questions
    df = df[df['text_type'] == 'q'].copy()

    # Nettoyage
    df['faq_clean'] = df['faq'].apply(preprocess)

    # Vérification des colonnes
    if 'faq' not in df.columns:
        st.error("❌ Le fichier CSV doit contenir la colonne 'faq'.")
        st.stop()

    return df

# === Fonction de similarité ===
def get_most_relevant_answer(df, user_input):
    user_input_clean = preprocess(user_input)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['faq_clean'].tolist() + [user_input_clean])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    index = similarity_scores.argmax()
    return df.iloc[index]['faq']

# === Fonction principale du chatbot ===
def chatbot(user_input, df):
    return get_most_relevant_answer(df, user_input)

# === Interface Streamlit ===
def main():
    st.title("🌍 Climate Change FAQ Chatbot")
    st.write("Posez une question sur le changement climatique et je vais chercher dans les FAQ officielles du GIEC/IPCC.")

    df = load_data()

    user_input = st.text_input("❓ Votre question :")

    if user_input:
        response = chatbot(user_input, df)
        st.markdown("💬 **Réponse la plus proche trouvée :**")
        st.write(response)

if __name__ == "__main__":
    main()
