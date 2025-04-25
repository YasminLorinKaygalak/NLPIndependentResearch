import math
import re
from collections import Counter
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def load_common_words(filepath: str) -> Dict[str, int]:
    """
    Load a list of common words with their frequencies into a dictionary.
    Format: [frequency word] per line.
    """
    freq_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    freq = int(parts[-2])
                    word = parts[-1].lower()
                    freq_dict[word] = freq
                except ValueError:
                    continue
    print(f"Loaded {len(freq_dict)} common words with frequencies.")
    return freq_dict


def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return word_tokenize(text)


def compute_entropy_using_common_vocab(tokens: List[str], common_freq: Dict[str, int]) -> Dict:
    total_common_freq = sum(common_freq.values())

    # Use input token frequencies for weighting
    token_counts = Counter(tokens)
    token_entropy_contrib = {}
    token_log_probs = {}

    total_tokens = sum(token_counts.values())
    entropy = 0.0

    for token, count in token_counts.items():
        token_lower = token.lower()
        freq_in_common = common_freq.get(token_lower, 1)  # Smoothing for unseen words
        if token_lower not in common_freq:
            print(token_lower)
        prob = freq_in_common / total_common_freq
        log_prob = -math.log2(prob)

        weight = count / total_tokens  # how often the word appears in the input
        entropy += weight * log_prob

        token_entropy_contrib[token] = weight * log_prob
        token_log_probs[token] = log_prob

    return {
        "Entropy": entropy,
        "Log Probabilities": token_log_probs,
        "Token Entropy Contribution": token_entropy_contrib
    }


def analyze_texts_against_common_vocab(texts: List[str], common_freq: Dict[str, int]) -> List[Dict]:
    results = []
    for text in texts:
        tokens = preprocess_text(text)
        analysis = compute_entropy_using_common_vocab(tokens, common_freq)
        results.append({
            "Text": text,
            "Entropy": analysis["Entropy"],
            "Log Probabilities": analysis["Log Probabilities"],
            "Token Entropy Contribution": analysis["Token Entropy Contribution"]
        })
    return results


# Example usage
if __name__ == "__main__":
    common_word_freq = load_common_words("common_words.txt")

    texts = [
        "Artificial intelligence is revolutionizing the modern world.",
        "The the the the the the the the the the.",
        "In this instance, since all soldiers rushed to the city..."
    ]

    results = analyze_texts_against_common_vocab(texts, common_word_freq)

    for res in results:
        print(f"\nText: {res['Text'][:60]}...")
        print(f"Entropy: {res['Entropy']:.4f}")
        print("Top Log Probabilities:")
        for word, logp in sorted(res["Log Probabilities"].items(), key=lambda x: -x[1])[:5]:
            print(f"  {word}: {logp:.4f}")
