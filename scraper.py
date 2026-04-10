import string
import requests
from bs4 import BeautifulSoup as bs
import csv
import time
import random
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
]

SEARCH_URLS = [
    "https://www.amazon.in/s?k=mobile",
    "https://www.amazon.in/s?k=earbuds",
]

BASE_URL = "https://www.amazon.in"

# ==========================================
# SCRAPING & CLEANING FUNCTIONS
# ==========================================
def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

def get_page(url, session):
    try:
        time.sleep(random.uniform(3, 6)) 
        r = session.get(url, headers=get_headers(), timeout=15)
        return r.text if r.status_code == 200 else None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def get_product_links(search_url, session):
    html = get_page(search_url, session)
    if not html: return []
    soup = bs(html, "html.parser")
    links = []
    for tag in soup.find_all("a", class_="a-link-normal"):
        href = tag.get("href", "")
        if "/dp/" in href:
            full_url = BASE_URL + href if not href.startswith("http") else href
            links.append(full_url)
    return list(set(links))

def get_reviews(product_url, session):
    print(f"Fetching reviews from: {product_url}")
    html = get_page(product_url, session)
    if not html: return []
    soup = bs(html, "html.parser")
    reviews = []
    for div in soup.find_all("div", {"class": "a-section celwidget"}):
        try:
            rating = div.find("i", {"data-hook": "review-star-rating"}).find("span", class_="a-icon-alt").text
            title = div.find("a", {"data-hook": "review-title"}).find_all("span")[-1].text
            body = div.find("span", {"data-hook": "review-body"}).text
            reviews.append((rating, title, body))
        except:
            continue
    return reviews

def preprocess_text(text):
    stop_words = {'a', 'an', 'the', 'and', 'is', 'it', 'in', 'on', 'for', 'with', 'was', 'of', 'to', 'this', 'that'}
    if not text: return ""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def clean_reviews_csv(input_file, output_file):
    cleaned_data = []
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 3: continue
                cleaned_data.append([row[0], preprocess_text(row[1]), preprocess_text(row[2])])
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(cleaned_data)
        print(f"Successfully cleaned data -> {os.path.abspath(output_file)}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")

# ==========================================
# ANALYSIS CLASSES
# ==========================================
class DocumentTermMatrix:
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus = self._load_corpus()
        self.vectorizer = CountVectorizer()
        self.dtm = self.vectorizer.fit_transform(self.corpus)

    def _load_corpus(self):
        corpus = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3: corpus.append(f"{row[1]} {row[2]}")
        return corpus

    def print_analysis(self, top_n=20):
        word_counts = np.asarray(self.dtm.sum(axis=0)).flatten()
        words = self.vectorizer.get_feature_names_out()
        word_freq = sorted(zip(words, word_counts), key=lambda x: x[1], reverse=True)
        print(f"\n--- Top {top_n} Frequent Words ---")
        for word, count in word_freq[:top_n]:
            print(f"{word:<15} | {int(count)}")

    def save_top_words(self, folder, filename, top_n=20):
        # Combine folder path and filename
        full_path = os.path.join(folder, filename)
        word_counts = np.asarray(self.dtm.sum(axis=0)).flatten()
        words = self.vectorizer.get_feature_names_out()
        word_freq = sorted(zip(words, word_counts), key=lambda x: x[1], reverse=True)
        with open(full_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Word", "Frequency"])
            writer.writerows(word_freq[:top_n])
        print(f"FILE SAVED: {os.path.abspath(full_path)}")


class OneHotDocumentVector:
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus = self._load_corpus()
        self.vectorizer = CountVectorizer(binary=True)
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def _load_corpus(self):
        corpus = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3: corpus.append(f"{row[1]} {row[2]}")
        return corpus

    def save_binary_matrix(self, folder, filename):
        full_path = os.path.join(folder, filename)
        df = pd.DataFrame(self.matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        df.to_csv(full_path, index=False)
        print(f"FILE SAVED: {os.path.abspath(full_path)}")

class TfidfAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus = self._load_corpus()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def _load_corpus(self):
        corpus = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3: corpus.append(f"{row[1]} {row[2]}")
        return corpus

    def save_tfidf_matrix(self, folder, filename):
        full_path = os.path.join(folder, filename)
        df = pd.DataFrame(self.matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        df.to_csv(full_path, index=False)
        print(f"FILE SAVED: {os.path.abspath(full_path)}")


# ==========================================
# UPDATED MAIN FUNCTION
# ==========================================
def main():
    # Create a dedicated folder for all outputs so they aren't lost
    OUTPUT_FOLDER = "nlp_results"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    RAW_FILE = os.path.join(OUTPUT_FOLDER, "amazon_reviews.csv")
    CLEANED_FILE = os.path.join(OUTPUT_FOLDER, "amazon_reviews_cleaned.csv")
    
    # STEP 1: SCRAPE
    session = requests.Session()
    all_reviews = []
    
    for s_url in SEARCH_URLS:
        print(f"Searching: {s_url}")
        links = get_product_links(s_url, session)
        for link in links[:5]: 
            all_reviews.extend(get_reviews(link, session))

    if all_reviews:
        with open(RAW_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rating", "Title", "Body"])
            writer.writerows(all_reviews)
        print(f"Raw data saved to {os.path.abspath(RAW_FILE)}")
        
        # STEP 2: CLEAN
        # Note: Modified clean_reviews_csv to accept paths
        clean_reviews_csv(RAW_FILE, CLEANED_FILE)

        # STEP 3: ANALYZE
        try:
            # Case 1: Bag of Words
            print("\n--- Processing Bag of Words ---")
            dtm_analyzer = DocumentTermMatrix(CLEANED_FILE)
            dtm_analyzer.print_analysis(top_n=15)
            dtm_analyzer.save_top_words(OUTPUT_FOLDER, "top_frequent_words.csv", top_n=15)
            
            # Case 2: One Hot Encoding
            print("\n--- Processing One Hot Encoding ---")
            ohe_analyzer = OneHotDocumentVector(CLEANED_FILE)
            ohe_analyzer.save_binary_matrix(OUTPUT_FOLDER, "one_hot_matrix.csv")
            
            # Case 3: TF-IDF
            print("\n--- Processing TF-IDF ---")
            tfidf_analyzer = TfidfAnalyzer(CLEANED_FILE)
            tfidf_analyzer.save_tfidf_matrix(OUTPUT_FOLDER, "tfidf_matrix.csv")
            
            print("\n" + "="*50)
            print(f"ALL FILES ARE LOCATED IN: {os.path.abspath(OUTPUT_FOLDER)}")
            print("="*50)

        except Exception as e:
            print(f"Analysis failed: {e}")      
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
