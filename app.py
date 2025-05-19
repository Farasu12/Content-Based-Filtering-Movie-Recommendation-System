import mysql.connector
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import numpy as np
import logging
import math

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'movie_recommendation'
}

# Global variable for caching
processed_dataset_cache = None

# Inisialisasi kamus bahasa Inggris dari NLTK
english_words = set(words.words())

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'#[^\s]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Fungsi untuk validasi kata kunci
def validate_keyword(keyword):
    if not keyword.strip():
        return "The keyword field cannot be empty"
    words = keyword.split()
    if len(words) > 5:  
        return "Keywords must not exceed 5 words"
    if not re.match(r'^[a-zA-Z\s]+$', keyword):
        return "Keywords must not contain numbers or symbols"
    keyword_words = keyword.lower().split()
    for word in keyword_words:
        if word not in english_words:
            return "Keywords must be in English only"
    return None

# Fungsi untuk menghitung NDCG@5
def dcg_at_k(relevance_scores, k=5):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))

def ndcg_at_k(relevance_scores, true_relevance, k=5):
    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k(sorted(true_relevance, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0

# Fungsi untuk menghitung MRR
def mrr(recommended_films, relevant_films):
    for rank, film in enumerate(recommended_films, 1):
        if film in relevant_films:
            return 1 / rank
    return 0

# Fungsi untuk menghitung Hit Rate
def hit_rate(recommended_films, relevant_films):
    return 1 if len(set(recommended_films) & relevant_films) > 0 else 0

# Content-Based Filtering function
def content_based_filtering(keyword):
    global processed_dataset_cache
    start_time = time.time()
    
    # 1. Ambil dataset yang sudah diproses dari tabel movie_processed
    conn = mysql.connector.connect(**db_config)
    if processed_dataset_cache is None:
        logging.info("Caching processed dataset for the first time.")
        processed_dataset_cache = pd.read_sql_query("SELECT * FROM movie_processed", conn)
    
    original_dataset = pd.read_sql_query("SELECT * FROM movie", conn)
    conn.close()
    
    # 2. Tentukan relevant films berdasarkan combined_text (tanpa ground truth)
    keyword_processed = preprocess_text(keyword)
    keyword_tokens = set(keyword_processed.split())
    relevant_films = set()
    for idx, row in processed_dataset_cache.iterrows():
        combined_tokens = set(str(row['combined_text']).lower().split())
        # Kecocokan token yang persis
        if keyword_tokens & combined_tokens:
            relevant_films.add(row['names'])
        # Tambahkan kecocokan parsial (minimal 30% token cocok)
        if len(keyword_tokens) > 0 and len(keyword_tokens & combined_tokens) / len(keyword_tokens) >= 0.3:
            relevant_films.add(row['names'])
    
    # 3. Preprocessing kata kunci
    keyword_list = keyword_processed.split()
    
    # Step 1: Calculate document lengths
    doc_lengths = [len(text.split()) for text in processed_dataset_cache['combined_text']]
    
    # Step 2: Calculate Term Frequency (TF)
    tf_matrix = np.zeros((len(processed_dataset_cache), len(keyword_list)))
    for i, text in enumerate(processed_dataset_cache['combined_text']):
        words = text.split()
        doc_length = doc_lengths[i]
        for j, term in enumerate(keyword_list):
            term_count = words.count(term)
            tf_matrix[i, j] = term_count / doc_length if doc_length > 0 else 0
    
    # Step 3: Calculate Inverse Document Frequency (IDF)
    total_docs = len(processed_dataset_cache)
    idf_values = [math.log(total_docs / (sum(1 for text in processed_dataset_cache['combined_text'] if term in text.split()) or 1)) for term in keyword_list]
    
    # Step 4: Calculate TF-IDF
    tfidf_matrix = np.zeros((len(processed_dataset_cache), len(keyword_list)))
    for i in range(len(processed_dataset_cache)):
        for j in range(len(keyword_list)):
            tfidf_matrix[i, j] = tf_matrix[i, j] * idf_values[j]
    
    # Batasi TF, IDF, dan TF-IDF ke 50 data terbesar
    tf_df = pd.DataFrame(tf_matrix, columns=keyword_list)
    tf_df.insert(0, 'Movie', processed_dataset_cache['names'].values)
    tf_df['Max_Abs_TF'] = tf_df[keyword_list].abs().max(axis=1)
    tf_df = tf_df.sort_values(by='Max_Abs_TF', ascending=False).head(50).drop(columns=['Max_Abs_TF'])

    idf_df = pd.DataFrame({'Term': keyword_list, 'IDF': idf_values})
    idf_df = idf_df.sort_values(by='IDF', ascending=False).head(50)

    tfidf_df = pd.DataFrame(tfidf_matrix, columns=keyword_list)
    tfidf_df.insert(0, 'Movie', processed_dataset_cache['names'].values)
    tfidf_df['Max_Abs_TF-IDF'] = tfidf_df[keyword_list].abs().max(axis=1)
    tfidf_df = tfidf_df.sort_values(by='Max_Abs_TF-IDF', ascending=False).head(50).drop(columns=['Max_Abs_TF-IDF'])

    # Step 5: Create query vector
    query_vector = np.array(idf_values)
    
    # Step 6: Calculate Cosine Similarity
    cosine_sim = np.zeros(len(processed_dataset_cache))
    for i in range(len(processed_dataset_cache)):
        doc_vector = tfidf_matrix[i]
        dot_product = np.sum(query_vector * doc_vector)
        query_magnitude = np.sqrt(np.sum(query_vector ** 2))
        doc_magnitude = np.sqrt(np.sum(doc_vector ** 2))
        if query_magnitude > 0 and doc_magnitude > 0:
            cosine_sim[i] = dot_product / (query_magnitude * doc_magnitude)
        else:
            cosine_sim[i] = 0
    
    # Step 7: Prepare Result Tables
    result_df = processed_dataset_cache.copy()
    result_df['cosine_score'] = cosine_sim
    
    # Step 8: Sort results
    result_df['score'] = pd.to_numeric(result_df['score'], errors='coerce')
    result_df['year'] = pd.to_numeric(result_df['year'], errors='coerce')
    result_df['final_score'] = (0.6 * result_df['cosine_score']) + \
                               (0.3 * (result_df['score'] / 10)) + \
                               (0.1 * (result_df['year'] / 2025))
    result_df = result_df.sort_values(by='final_score', ascending=False)
    
    # Get top 5 recommendations
    recommendations = result_df.head(5)
    
    # Batasi Cosine Similarity Table ke 50 data terbesar
    cosine_df = result_df[['names', 'overview', 'genre', 'crew', 'score', 'year', 'cosine_score', 'poster_url']].copy()
    cosine_df = cosine_df.sort_values(by='cosine_score', ascending=False).head(50)

    # Calculate evaluation metrics
    recommended_films = set(recommendations['names'])
    recommended_films_list = list(recommendations['names'])
    
    # Calculate NDCG@5
    relevance_scores = []
    true_relevance = []
    for film in recommended_films_list:
        if film in relevant_films:
            idx = processed_dataset_cache[processed_dataset_cache['names'] == film].index[0]
            cosine_score = cosine_sim[idx]
            # Skala relevansi yang lebih ketat
            if cosine_score > 0.7:
                relevance_scores.append(3)
                true_relevance.append(3)
            elif cosine_score > 0.4:
                relevance_scores.append(2)
                true_relevance.append(2)
            else:
                relevance_scores.append(1)
                true_relevance.append(1)
        else:
            relevance_scores.append(0)
            true_relevance.append(0)
    ndcg_score = ndcg_at_k(relevance_scores, true_relevance)

    # Calculate MRR
    mrr_score = mrr(recommended_films_list, relevant_films)

    # Calculate Hit Rate
    hit_rate_score = hit_rate(recommended_films, relevant_films)

    # Prepare processed_dataset with additional columns for display
    processed_dataset = processed_dataset_cache.copy()
    processed_dataset['word_count'] = processed_dataset['combined_text'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Log execution time and evaluation metrics
    execution_time = time.time() - start_time
    logging.info(f"Execution time for keyword '{keyword}': {execution_time} seconds")
    logging.info(f"Evaluation for keyword '{keyword}': NDCG@5={ndcg_score}, MRR={mrr_score}, Hit Rate={hit_rate_score}")
    
    return {
        'original_dataset': original_dataset,
        'processed_dataset': processed_dataset,
        'tf_df': tf_df,
        'idf_df': idf_df,
        'tfidf_df': tfidf_df,
        'cosine_df': cosine_df,
        'recommendations': recommendations,
        'ndcg_score': ndcg_score,
        'mrr_score': mrr_score,
        'hit_rate': hit_rate_score,
        'keywords': keyword_list
    }

# Initialize database with movie data
def init_db():
    global processed_dataset_cache
    conn = mysql.connector.connect(**db_config)
    c = conn.cursor()
    
    # Buat tabel movie jika belum ada
    c.execute("""
        CREATE TABLE IF NOT EXISTS movie (
            id INT AUTO_INCREMENT PRIMARY KEY,
            names VARCHAR(255) NOT NULL,
            year INT,
            score FLOAT,
            genre TEXT,
            overview TEXT,
            orig_lang VARCHAR(50),
            crew TEXT,
            country VARCHAR(100),
            poster_url TEXT  -- Tambahkan kolom poster_url
        )
    """)
    
    # Buat tabel movie_processed untuk menyimpan data yang sudah diproses
    c.execute("""
        CREATE TABLE IF NOT EXISTS movie_processed (
            id INT AUTO_INCREMENT PRIMARY KEY,
            names VARCHAR(255) NOT NULL,
            year INT,
            score FLOAT,
            genre TEXT,
            overview TEXT,
            orig_lang VARCHAR(50),
            crew TEXT,
            country VARCHAR(100),
            combined_text TEXT NOT NULL,
            poster_url TEXT  -- Tambahkan kolom poster_url
        )
    """)
    
    # Kosongkan kedua tabel sebelum inisialisasi
    c.execute("TRUNCATE TABLE movie")
    c.execute("TRUNCATE TABLE movie_processed")
    conn.commit()
    print("Cleared tables 'movie' and 'movie_processed' before initialization.")
    
    # Reset cache
    processed_dataset_cache = None
    logging.info("Cleared cache during database initialization.")
    
    # Load dataset baru
    try:
        df = pd.read_csv('imdb_filtered_1000.csv')
    except FileNotFoundError:
        logging.error("File 'imdb_filtered_1000.csv' not found. Database initialization skipped.")
        print("Error: File 'imdb_filtered_1000.csv' not found. Please ensure the file exists in the project directory.")
        conn.close()
        return
    
    # Preprocessing setiap kolom
    df['names'] = df['names'].apply(preprocess_text)
    df['overview'] = df['overview'].apply(preprocess_text)
    df['genre'] = df['genre'].apply(preprocess_text)
    df['crew'] = df['crew'].apply(preprocess_text)
    
    # Hapus duplikat berdasarkan kolom 'names'
    df = df.drop_duplicates(subset=['names'], keep='first')
    print(f"Total movies after removing duplicates: {len(df)}")
    logging.info(f"Total movies after removing duplicates: {len(df)}")
    
    # Gabungkan kolom names, overview, genre, dan crew menjadi satu teks panjang
    df['combined_text'] = df['names'].fillna('') + ' ' + \
                         df['overview'].fillna('') + ' ' + \
                         df['genre'].fillna('') + ' ' + \
                         df['crew'].fillna('')
    
    # Simpan data asli ke tabel movie, termasuk poster_url
    for _, row in df.iterrows():
        c.execute("""
            INSERT INTO movie (names, year, score, genre, overview, orig_lang, crew, country, poster_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (row['names'], row['year'], row['score'], row['genre'], row['overview'], 
              row['orig_lang'], row['crew'], row['country'], row['poster_url']))
    
    # Simpan data yang sudah diproses ke tabel movie_processed, termasuk poster_url
    for _, row in df.iterrows():
        c.execute("""
            INSERT INTO movie_processed (names, year, score, genre, overview, orig_lang, crew, country, combined_text, poster_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (row['names'], row['year'], row['score'], row['genre'], row['overview'], 
              row['orig_lang'], row['crew'], row['country'], row['combined_text'], row['poster_url']))
    
    conn.commit()
    conn.close()
    print("Database initialization completed.")
    logging.info("Database initialized with new dataset.")

@app.route('/')
def index():
    return render_template('index.html', recommendations=None)

@app.route('/recommend', methods=['POST'])
def recommend():
    keyword = request.form['keyword']
    
    # Validasi kata kunci
    error = validate_keyword(keyword)
    if error:
        logging.warning(f"Invalid keyword submitted by user: '{keyword}' - {error}")
        return render_template('index.html', error=error, recommendations=None)
    
    # Log aktivitas pengguna
    logging.info(f"User submitted recommendation request with keyword: '{keyword}'")
    
    # Panggil content_based_filtering
    result = content_based_filtering(keyword)
    recommendations = result['recommendations']
    
    # Pop success message to display it once
    recommendation_success = session.pop('recommendation_success', None)
    logging.info(f"Index route: recommendation_success = {recommendation_success}")
    
    # Set success message
    session['recommendation_success'] = f"Successfully generated recommendations for keyword: '{keyword}'"
    
    return render_template('index.html', recommendations=recommendations, keyword=keyword)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = mysql.connector.connect(**db_config)
        c = conn.cursor()
        c.execute("SELECT * FROM admin WHERE username = %s AND password = %s", (username, password))
        admin = c.fetchone()
        conn.close()
        
        if admin:
            session['admin_id'] = admin[0]
            session['login_success'] = f"Admin '{username}' logged in successfully!"  # Set login success message
            logging.info(f"Admin '{username}' logged in successfully.")
            return redirect(url_for('admin_home'))
        else:
            logging.warning(f"Failed login attempt for admin '{username}'.")
            return render_template('login.html', error="Invalid username or password.")
    
    return render_template('login.html')

@app.route('/admin_home', methods=['GET', 'POST'])
def admin_home():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    
    error = None
    warning = session.pop('warning', None)
    
    if request.method == 'POST':
        keyword = request.form['keyword']
        
        # Validasi kata kunci
        error = validate_keyword(keyword)
        if error:
            logging.warning(f"Invalid keyword submitted by admin: '{keyword}' - {error}")
            return render_template('admin_home.html', error=error, warning=warning)
        
        # Simpan kata kunci di session dan tandai bahwa pencarian telah dilakukan
        session['has_searched'] = True
        session['last_keyword'] = keyword
        logging.info(f"Admin submitted valid keyword: '{keyword}'")
        
        return redirect(url_for('cbf', keyword=keyword))
    
    return render_template('admin_home.html', error=error, warning=warning)

@app.route('/admin/dataset', methods=['GET', 'POST'], strict_slashes=False)
def dataset():
    global processed_dataset_cache
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    
    error = None
    warning = session.pop('warning', None)
    
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if not file or not file.filename.endswith('.csv'):
                error = "File must be in CSV format."
                logging.warning("Upload attempt with non-CSV file.")
                try:
                    conn = mysql.connector.connect(**db_config)
                    movies = pd.read_sql_query("SELECT * FROM movie", conn)
                    conn.close()
                    logging.info(f"Fetched {len(movies)} records from the 'movie' table after failed upload.")
                except mysql.connector.Error as e:
                    logging.error(f"Error fetching data from 'movie' table: {e}")
                    movies = pd.DataFrame()
                return render_template('dataset.html', error=error, movies=movies, 
                                     upload_success=None, delete_success=None, warning=warning)
            
            try:
                # Baca file CSV tanpa menghapus dataset yang ada
                df = pd.read_csv(file)
                
                # Kolom wajib yang harus ada
                required_columns = ['names', 'year', 'score', 'genre', 'overview', 
                                  'orig_lang', 'crew', 'country', 'poster_url']
                
                # Validasi apakah semua kolom wajib ada
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    error = f"Missing required columns: {', '.join(missing_columns)}"
                    logging.warning(f"Dataset upload failed: {error}")
                    try:
                        conn = mysql.connector.connect(**db_config)
                        movies = pd.read_sql_query("SELECT * FROM movie", conn)
                        conn.close()
                        logging.info(f"Fetched {len(movies)} records from the 'movie' table after failed upload.")
                    except mysql.connector.Error as e:
                        logging.error(f"Error fetching data from 'movie' table: {e}")
                        movies = pd.DataFrame()
                    return render_template('dataset.html', error=error, movies=movies, 
                                         upload_success=None, delete_success=None, warning=warning)
                
                # Validasi tipe data untuk year dan score
                try:
                    df['year'] = pd.to_numeric(df['year'], errors='raise')
                    df['score'] = pd.to_numeric(df['score'], errors='raise')
                except ValueError:
                    error = "Columns 'year' and 'score' must contain numeric values."
                    logging.warning(f"Dataset upload failed: {error}")
                    try:
                        conn = mysql.connector.connect(**db_config)
                        movies = pd.read_sql_query("SELECT * FROM movie", conn)
                        conn.close()
                        logging.info(f"Fetched {len(movies)} records from the 'movie' table after failed upload.")
                    except mysql.connector.Error as e:
                        logging.error(f"Error fetching data from 'movie' table: {e}")
                        movies = pd.DataFrame()
                    return render_template('dataset.html', error=error, movies=movies, 
                                         upload_success=None, delete_success=None, warning=warning)
                
                # Jika semua validasi berhasil, lanjutkan dengan preprocessing dan simpan dataset
                # Preprocessing setiap kolom
                df['names'] = df['names'].apply(preprocess_text)
                df['overview'] = df['overview'].apply(preprocess_text)
                df['genre'] = df['genre'].apply(preprocess_text)
                df['crew'] = df['crew'].apply(preprocess_text)
                
                # Hapus duplikat berdasarkan kolom 'names'
                df = df.drop_duplicates(subset=['names'], keep='first')
                
                # Gabungkan kolom names, overview, genre, dan crew menjadi satu teks panjang
                # Mengisi nilai kosong dengan string kosong untuk mencegah error saat penggabungan
                df['combined_text'] = df['names'].fillna('') + ' ' + \
                                     df['overview'].fillna('') + ' ' + \
                                     df['genre'].fillna('') + ' ' + \
                                     df['crew'].fillna('')
                
                # Sekarang hapus dataset lama dan simpan yang baru
                conn = mysql.connector.connect(**db_config)
                c = conn.cursor()
                
                # Kosongkan kedua tabel hanya jika validasi berhasil
                c.execute("TRUNCATE TABLE movie")
                c.execute("TRUNCATE TABLE movie_processed")
                
                # Reset cache
                processed_dataset_cache = None
                logging.info("Cleared cache after uploading new dataset.")
                
                # Simpan data asli ke tabel movie, termasuk poster_url
                for _, row in df.iterrows():
                    c.execute("""
                        INSERT INTO movie (names, year, score, genre, overview, orig_lang, crew, country, poster_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (row['names'], row['year'], row['score'], row['genre'], row['overview'], 
                          row['orig_lang'], row['crew'], row['country'], row.get('poster_url', None)))
                
                # Simpan data yang sudah diproses ke tabel movie_processed, termasuk poster_url
                for _, row in df.iterrows():
                    c.execute("""
                        INSERT INTO movie_processed (names, year, score, genre, overview, orig_lang, crew, country, combined_text, poster_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (row['names'], row['year'], row['score'], row['genre'], row['overview'], 
                          row['orig_lang'], row['crew'], row['country'], row['combined_text'], row.get('poster_url', None)))
                
                conn.commit()
                conn.close()
                logging.info("Admin uploaded a new dataset.")
                session['upload_success'] = "New dataset uploaded successfully!"
                
                # Ambil dataset yang baru diupload untuk ditampilkan
                try:
                    conn = mysql.connector.connect(**db_config)
                    movies = pd.read_sql_query("SELECT * FROM movie", conn)
                    conn.close()
                    logging.info(f"Fetched {len(movies)} records from the 'movie' table after successful upload.")
                except mysql.connector.Error as e:
                    logging.error(f"Error fetching data from 'movie' table: {e}")
                    movies = pd.DataFrame()
                
                return render_template('dataset.html', error=None, movies=movies, 
                                     upload_success=session['upload_success'], delete_success=None, warning=warning)
            
            except Exception as e:
                error = f"Error processing file: {str(e)}"
                logging.error(f"Dataset upload failed: {error}")
                try:
                    conn = mysql.connector.connect(**db_config)
                    movies = pd.read_sql_query("SELECT * FROM movie", conn)
                    conn.close()
                    logging.info(f"Fetched {len(movies)} records from the 'movie' table after failed upload.")
                except mysql.connector.Error as e:
                    logging.error(f"Error fetching data from 'movie' table: {e}")
                    movies = pd.DataFrame()
                return render_template('dataset.html', error=error, movies=movies, 
                                     upload_success=None, delete_success=None, warning=warning)
        
        elif 'delete' in request.form:
            conn = mysql.connector.connect(**db_config)
            c = conn.cursor()
            c.execute("TRUNCATE TABLE movie")
            c.execute("TRUNCATE TABLE movie_processed")
            
            # Reset cache
            processed_dataset_cache = None
            logging.info("Cleared cache after deleting dataset.")
            
            conn.commit()
            conn.close()
            logging.info("Admin deleted the dataset.")
            session['delete_success'] = "Dataset deleted successfully!"
            return render_template('dataset.html', error=None, movies=pd.DataFrame(), 
                                 upload_success=None, delete_success=session['delete_success'], warning=warning)
    
    try:
        conn = mysql.connector.connect(**db_config)
        movies = pd.read_sql_query("SELECT * FROM movie", conn)
        conn.close()
        logging.info(f"Fetched {len(movies)} records from the 'movie' table.")
    except mysql.connector.Error as e:
        logging.error(f"Error fetching data from 'movie' table: {e}")
        movies = pd.DataFrame()
    
    # Pop success messages to display them once
    upload_success = session.pop('upload_success', None)
    delete_success = session.pop('delete_success', None)
    
    return render_template('dataset.html', error=None, movies=movies, 
                         upload_success=upload_success, delete_success=delete_success, warning=warning)

@app.route('/admin/cbf', methods=['GET', 'POST'], strict_slashes=False)
def cbf():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    
    if not session.get('has_searched', False):
        session['warning'] = "Please fill in keywords and do a search first"
        return redirect(url_for('admin_home'))
    
    error = None
    keyword = None
    
    if request.method == 'POST':
        keyword = request.form['keyword']
    elif request.method == 'GET' and 'keyword' in request.args:
        keyword = request.args.get('keyword')
    
    if keyword:
        # Validasi kata kunci
        error = validate_keyword(keyword)
        if error:
            logging.warning(f"Invalid keyword submitted by admin: '{keyword}' - {error}")
            return render_template('cbf.html', error=error)
        
        # Log aktivitas admin
        logging.info(f"Admin requested CBF with keyword: '{keyword}'")
        
        # Simpan kata kunci di session dan tandai bahwa pencarian telah dilakukan
        session['has_searched'] = True
        session['last_keyword'] = keyword
        
        # Panggil content_based_filtering
        result = content_based_filtering(keyword)
        
        # Konversi DataFrame ke list of dictionaries untuk template
        original_dataset = result['original_dataset'].to_dict('records')
        processed_dataset = result['processed_dataset'].to_dict('records')
        tf_df = result['tf_df'].to_dict('records')
        idf_df = result['idf_df'].to_dict('records')
        tfidf_df = result['tfidf_df'].to_dict('records')
        cosine_df = result['cosine_df'].to_dict('records')
        recommendations = result['recommendations'].to_dict('records')
        keywords = result['keywords']
        
        return render_template('cbf.html',
                              keyword=keyword,
                              original_dataset=original_dataset,
                              processed_dataset=processed_dataset,
                              tf_df=tf_df,
                              idf_df=idf_df,
                              tfidf_df=tfidf_df,
                              cosine_df=cosine_df,
                              recommendations=recommendations,
                              ndcg_score=result['ndcg_score'],
                              mrr_score=result['mrr_score'],
                              hit_rate=result['hit_rate'],
                              keywords=keywords)
    
    # Jika tidak ada kata kunci baru, gunakan kata kunci terakhir dari session
    if 'last_keyword' in session:
        keyword = session['last_keyword']
        result = content_based_filtering(keyword)
        
        # Konversi DataFrame ke list of dictionaries untuk template
        original_dataset = result['original_dataset'].to_dict('records')
        processed_dataset = result['processed_dataset'].to_dict('records')
        tf_df = result['tf_df'].to_dict('records')
        idf_df = result['idf_df'].to_dict('records')
        tfidf_df = result['tfidf_df'].to_dict('records')
        cosine_df = result['cosine_df'].to_dict('records')
        recommendations = result['recommendations'].to_dict('records')
        keywords = result['keywords']
        
        return render_template('cbf.html',
                              keyword=keyword,
                              original_dataset=original_dataset,
                              processed_dataset=processed_dataset,
                              tf_df=tf_df,
                              idf_df=idf_df,
                              tfidf_df=tfidf_df,
                              cosine_df=cosine_df,
                              recommendations=recommendations,
                              ndcg_score=result['ndcg_score'],
                              mrr_score=result['mrr_score'],
                              hit_rate=result['hit_rate'],
                              keywords=keywords)
    
    return render_template('cbf.html', error=error)

@app.route('/logout')
def logout():
    admin_id = session.get('admin_id')
    logging.info(f"Admin logged out. Session: {admin_id}")
    session.pop('has_searched', None)
    session.pop('last_keyword', None)
    session.pop('admin_id', None)
    session['logout_success'] = "You have successfully logged out!"  # Set logout success message
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)