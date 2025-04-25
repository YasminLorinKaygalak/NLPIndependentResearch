import nltk
from nltk.tokenize import word_tokenize
import string
import re

nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocess the input text by removing numbers, URLs, punctuation, and converting to lowercase.
    """
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.lower()  # Convert to lowercase

def calculate_mattr(tokens, window_size=50):
    """
    Compute the Moving-Average Type-Token Ratio (MATTR).
    :param tokens: List of tokens from the processed text.
    :param window_size: Size of the sliding window.
    :return: The average TTR over all sliding windows.
    """
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0  # Fall back to standard TTR if text is shorter than window

    ttr_values = [
        len(set(tokens[i:i + window_size])) / window_size
        for i in range(len(tokens) - window_size + 1)
    ]

    return sum(ttr_values) / len(ttr_values)

def analyze_mattr(texts, window_size=50):
    """
    Analyze the MATTR for multiple texts.
    :param texts: List of input texts to analyze.
    :param window_size: Size of the sliding window.
    :return: List of MATTR scores.
    """
    results = []
    for text in texts:
        processed_text = preprocess_text(text)
        tokens = word_tokenize(processed_text)
        mattr_score = calculate_mattr(tokens, window_size)
        results.append(mattr_score)
    return results

# Example usage
if __name__ == "__main__":
    texts = [
        "Engineers built viaducts across valleys with rivers flowing through them.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries worldwide."
    ]
    mattr_results = analyze_mattr(texts, window_size=50)

    for i, text in enumerate(texts):
        print(f"Text: {text}")
        print(f"MATTR Score: {mattr_results[i]:.4f}\n")
