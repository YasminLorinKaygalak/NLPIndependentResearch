import numpy as np
import matplotlib.pyplot as plt
import re
import spacy
from typing import List, Dict

def load_common_words(filepath: str) -> set:
    """
    Load a list of common words from a file into a set.
    Each line in the file should contain a word and its frequency, separated by whitespace.
    """
    common_words = set()
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if parts:  # Make sure the line is not empty
                word = parts[-1]  # Extract the word
                common_words.add(word.lower())
                
    print(f"Loaded {len(common_words)} common words.")  # Print how many words are loaded
    return common_words

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess the text by converting to lowercase and removing punctuation.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()

def calculate_burstiness(tokens: List[str], common_words: set) -> float:
    """
    Calculate the burstiness of rare words in a tokenized text.
    """
    rare_word_sequence = [1 if word not in common_words else 0 for word in tokens]

    mean_occurrence = np.mean(rare_word_sequence)
    std_deviation = np.std(rare_word_sequence)

    if mean_occurrence == 0:
        return -1  # To indicate that the burstiness calculation is not applicable
    else:
        return (std_deviation - mean_occurrence) / (std_deviation + mean_occurrence)

def analyze_texts(texts: List[str], common_words: set) -> Dict[str, float]:
    """
    Process multiple texts using spaCy's nlp.pipe and compute burstiness scores.
    """
    nlp = spacy.load("en_core_web_sm")
    results = {}
    
    for doc in nlp.pipe(texts, batch_size=10):
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        burstiness_score = calculate_burstiness(tokens, common_words)
        results[doc.text] = burstiness_score
    
    return results

def plot_burstiness(rare_word_sequence: List[int], title: str):
    """
    Plot the distribution of rare words across the text.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rare_word_sequence, color='blue')
    plt.title(f'Distribution of Rare Words - {title}')
    plt.xlabel('Word Position')
    plt.ylabel('Rare Word Indicator (1=Rare, 0=Common)')
    plt.show()

def main():
    # Load the list of common words
    common_words = load_common_words('common_words.txt')

    # Sample texts for analysis
    texts = [
        "In this instance, since all soldiers rushed straight to Wenchuan and no troops were immediately sent to badly hit Beichuan...",
        "Artificial intelligence is transforming industries worldwide.",
        "The quick brown fox jumps over the lazy dog."
    ]

    # Analyze texts and get burstiness scores
    burstiness_results = analyze_texts(texts, common_words)

    # Display results
    for text, burstiness in burstiness_results.items():
        print(f"Text: {text[:50]}...")  # Show first 50 characters
        print(f"Burstiness Score: {burstiness}")
        
        # Preprocess and plot for visualization
        tokens = preprocess_text(text)
        rare_word_sequence = [1 if word not in common_words else 0 for word in tokens]
        plot_burstiness(rare_word_sequence, title=text[:50])

if __name__ == "__main__":
    main()
