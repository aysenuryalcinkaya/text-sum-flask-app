from flask import Flask, render_template, request ,send_file
import json
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
app = Flask(__name__, static_folder='static')
results = []
nltk.download('punkt')
def split_text_into_paragraphs(text):
    return text.split('\n\n')  # Paragraflar arasında boş satır olmalı

def build_similarity_matrix(sentences):
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

def textrank_summary(sentences, num_to_mark):
    similarity_matrix = build_similarity_matrix(sentences)
    
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return [sentence for score, sentence in ranked_sentences[:num_to_mark]]

def shorten_sentences(sentences, max_words):
    shortened_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            shortened = ' '.join(words[:max_words])
        else:
            shortened = sentence
        shortened_sentences.append(shortened)
    return shortened_sentences


def text_summary(input_text, num_to_mark=6, max_words=40):
    paragraphs = split_text_into_paragraphs(input_text)
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))

    important_sentences = textrank_summary(sentences, num_to_mark=num_to_mark)
    shortened_sentences = shorten_sentences(important_sentences, max_words=max_words)

    summary = ' '.join(shortened_sentences)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def get_summary():
    if request.method == 'POST':
        text = request.form['text']
        
        summary = text_summary(text)
        
        result_dict = {'original_text': text, 'summary': summary}
        results.append(result_dict)
        
        return render_template('result.html', original_text=text, summary=summary)

@app.route('/save', methods=['POST'])
def save_result():
    if request.method == 'POST':
        summary_text = results[-1]['summary']  # Get the latest summary
        
        with open('summary.txt', 'w', encoding='utf-8') as file:
            file.write(summary_text)
        
        return send_file('summary.txt', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)