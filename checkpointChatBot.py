import streamlit as st
import random
import json
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Téléchargement des données NLTK
nltk.download('punkt')
nltk.download('stopwords')

# --- 1. Chargement des données RASA NLU ---
def charger_exemples(chemin):
    with open(chemin, 'r', encoding='utf-8') as fichier:
        data = json.load(fichier)
    return data['rasa_nlu_data']['common_examples']

# --- 2. Prétraitement ---
def pretraiter(texte):
    texte = texte.lower()
    texte = re.sub(r'\W+', ' ', texte)
    tokens = nltk.word_tokenize(texte)
    stop_words = set(stopwords.words('english'))  # ou 'french' si tes données sont en français
    return ' '.join([w for w in tokens if w not in stop_words])

# --- 3. Préparer le dataset ---
def preparer_ensemble(common_examples):
    texts = []
    intents = []
    for exemple in common_examples:
        texts.append(pretraiter(exemple["text"]))
        intents.append(exemple["intent"])
    return texts, intents

# --- 4. Trouver l’intent le plus proche ---
def trouver_intent(question, texts, intents, vectorizer):
    question_proc = pretraiter(question)
    question_vec = vectorizer.transform([question_proc])
    text_vecs = vectorizer.transform(texts)
    similarites = cosine_similarity(question_vec, text_vecs).flatten()
    if max(similarites) == 0:
        return None
    best_index = similarites.argmax()
    return intents[best_index]

# --- 5. Fonction du chatbot ---
def chatbot(question, intents_data, texts, intents, vectorizer):
    intent = trouver_intent(question, texts, intents, vectorizer)
    if not intent:
        return "🤖 Désolé, je ne comprends pas votre question."
    # Générer une réponse fictive
    return f"Intent reconnu : **{intent}** ✅"

# --- 6. Interface Streamlit ---
def main():
    st.set_page_config(page_title="Chatbot Rasa NLU", page_icon="🤖")
    st.title("🌦️ Chatbot basé sur des données Rasa NLU")
    st.markdown("Posez une question météo, et je vais détecter l'intention.")

    chemin_json = r"C:\Users\Waad RTIBI\checkpointChatBot\weather_intent_entities.json"
    examples = charger_exemples(chemin_json)
    global texts, intents, vectorizer
    texts, intents = preparer_ensemble(examples)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)

    question = st.text_input("Votre question :")
    if question:
        reponse = chatbot(question, examples, texts, intents, vectorizer)
        st.markdown(f"**Réponse du chatbot :** {reponse}")

if __name__ == "__main__":
    main()
