
# Przykład prostego systemu RAG (Retrieval-Augmented Generation) w Pythonie
# Wersja edukacyjna – bez OpenAI API, z lokalnym przeszukiwaniem dokumentów

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Korpus wiedzy (lokalna baza dokumentów)
documents = [
    "Python to język programowania ogólnego przeznaczenia.",
    "Sieci neuronowe są wykorzystywane w sztucznej inteligencji.",
    "Uczenie maszynowe to technika pozwalająca komputerom uczyć się z danych.",
    "Transformery to nowoczesna architektura sieci neuronowych.",
    "OpenAI stworzyło modele GPT do generowania tekstu."
]

# Zapytanie użytkownika
query = "Do czego służą transformery w AI?"

# Krok 1: Wektoryzacja dokumentów + zapytania
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents + [query])
query_vector = doc_vectors[-1]
doc_vectors = doc_vectors[:-1]

# Krok 2: Obliczanie podobieństwa kosinusowego
similarities = cosine_similarity(query_vector, doc_vectors)[0]
top_k = similarities.argsort()[::-1][:2]  # 2 najbardziej podobne dokumenty

# Krok 3: Budowanie odpowiedzi generatywnej
context = "\n".join([documents[i] for i in top_k])
generated_answer = f"""Pytanie: {query}
Odpowiedź na podstawie wiedzy:
{context}
Na podstawie powyższego kontekstu, transformery to nowoczesna architektura sieci neuronowych stosowana w AI, m.in. w modelach takich jak GPT."""

print(generated_answer)
