import re
import json
import os
import nltk
from collections import Counter
from nltk.corpus import stopwords
from math import log
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

# nltk.download('stopwords', download_dir='.') #Uncomment when downloading stopwords file locally
stop_words = set(stopwords.words('english'))
stop_words = {re.sub(r"[’']", "", w) for w in stop_words} # Remove apostrophes from stopwords
domain_words = {"mongodb", "database", "collection", "document", "cloud", "data", "hi", "hey", "yeah", "like", "uh", "zhi", "arjun", "acme", "corp", "atlas"} # Domain specific generic words
stop_words = stop_words | domain_words # combine to one list of stop_words

# global cache for idf memoization
idf_cache = {}  # global cache dictionary


def read_record(filename)->list:
    """Returns a list of all the words in the given transcript"""
    with open(filename, 'r') as file:
        data = json.load(file)
    wordlist =[]
    for sentence in data["transcription_sentences"]:
        wordlist += sentence['text'].split()
    return wordlist

def clean_data(wordlist)->list:
    """
    Takes in a list and returns a copy clean of:
    - Remove punctuation
    - Lowercase all words
    - Remove NLDK stop words and domain specific terminology
    """
    cleaned = []
    for word in wordlist:
        word = word.lower()
        # Used GPT for help formatting RegEx
        word = re.sub(r'[—–-]', ' ', word) # Split em dashes and dashes into separated words
        tokens = word.split() # Handles words that were previously combined with dashes 
        for token in tokens:
                token = re.sub(r'[^\w\s]', '', token)  # Removes punctuation
                # Filter out stopwords and empty strings
                if token and token not in stop_words:
                    cleaned.append(token)
    return cleaned

def get_tf(word, record)->float:
    return record["counts"][word] / len(record["text"])

def get_idf(word, records)->float:
    """
    Computes IDF for given word using dictionary of all transcripts uses
    """
    n = len(records)
    df = 0
    if word in idf_cache:
        return idf_cache[word]

    # Iterate over the records and count words frequency
    for i in range(n):
        for key in records[f"transcript{i+1}.json"]["counts"].keys():
            if key == word:
                df += 1        
    
    value = log((n + 1) / (df + 1)) + 1.0 # IDF equation
    idf_cache[word] = value  # cache result
    return value
    
def get_high_tfidf(record, records):
    """
    This function returns the 3 words in the given record with the highest TFIDF
    """
    tfidf = {}
    for word in set(record['text']):
        if word not in tfidf:
            tfidf[word] = get_tf(word,record) * get_idf(word,records)

    # Used GPT to assist to get descending sorted formatting
    sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

    # Return the top 3 as (word, score) tuples
    return sorted_tfidf[:3]
    

def generate_wordclouds(clean_transcripts: dict, output_dir="outputs"):
    """
    Creates and saves a word cloud image for each transcript based on its cleaned text.
    Saves PNGs under the given output_dir. Used GPT to help me format and create the WC PNGs
    """

    for filename, data in clean_transcripts.items():
        # Combine cleaned words into one string
        text = " ".join(data["text"])

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100
        ).generate(text)

        # Save to outputs/
        out_path = os.path.join(output_dir, filename.replace(".json", "_wordcloud.png"))
        wc.to_file(out_path)
        print(f"Saved wordcloud → {out_path}")

def text_similarity_and_mds(clean_transcripts):
    """
    Build a cosine-similarity matrix from the word counts and visualizes the relationships between documents using MDS.
    GPT used to help answer Q&A about SKLearn functions.
    """
    filenames = sorted(clean_transcripts.keys())
    
    # Use DictVectorizer to convert list of {term: count} dicts into a "CSR matrix" 
    dicts = [clean_transcripts[fn]["counts"] for fn in filenames] # Used GPT to correct this list comprehension. I wrote it wrong the first time :(
    X = DictVectorizer(sparse=True).fit_transform(dicts)

    D = pairwise_distances(X, metric='cosine')
    coord = MDS(dissimilarity='precomputed', random_state=42).fit_transform(D)

    # Build and output the plot 
    plt.scatter(coord[:, 0], coord[:, 1])
    for i, fn in enumerate(filenames):
        plt.annotate(fn.replace(".json", ""), (coord[i, 0], coord[i, 1]))
    plt.title("Text Similarity Clustering (MDS)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()


def main():
    # Create dictionaries with wordlists for the transcripts 
    transcripts = {}
    for filename in os.listdir('data'):
        transcripts[filename] = read_record(f"data/{filename}")
    
    # Create dictionaries with cleaned transcript and unique word count
    clean_transcripts = {}
    for key, value in transcripts.items():
        text = clean_data(value)  # Calls the cleaning function
        counts = Counter(text)  # Unique count of cleaned words
        counts = dict(counts.most_common()) # Sorts descendingly
        clean_transcripts[key] = {
            "text": text,
            "counts": counts
        }
    
    # Print Analysis of each document
    for i in range(len(clean_transcripts)):
        print(f"Summary of Transcript {i+1}: \n")
        record = clean_transcripts[f"transcript{i+1}.json"]
        
        # Print Top 10 most common words
        counts = record["counts"]
        top10 = list(counts.items())[:10]
        print(f"Most common words:")
        j = 1
        for word, freq in top10:
            print(f"{j}) {word}: {freq}")
            j += 1
        print()

        # Compute average word length
        wordlength = 0
        for word in record["text"]:
            wordlength += len(word)
        wordlength = wordlength/len(record["text"])
        print(f"The average length of a word is: {wordlength:.2f}\n")
        
        tfidf = get_high_tfidf(record, clean_transcripts)
        print("The Highest TF-IDFs are:")
        j = 1
        for key, value in tfidf:
            print(f"{j}) {key}: {value:.3f}")
            j += 1
            
        print("")

    # generate_wordclouds(clean_transcripts, output_dir="outputs") # Commented out for testing
    text_similarity_and_mds(clean_transcripts)
    

if __name__ == "__main__":
    main()
