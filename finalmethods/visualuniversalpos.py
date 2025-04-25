import spacy
from collections import Counter
import matplotlib.pyplot as plt
from spacy import displacy

def calculate_pos_frequencies(filepath):
    nlp = spacy.load("en_core_web_sm")

    pos_counter = Counter()
    total_tokens = 0
    count = 0

    with open(filepath, 'r', encoding='utf-8') as input_file:
        text = input_file.read()

    lines = text.split('\n')
    docs = []

    for line in lines:
        if not line.strip():
            continue
        count += 1
        doc = nlp(line)
        docs.append(doc)
        pos_counter.update([token.pos_ for token in doc])
        total_tokens += len(doc)
        if count % 100 == 0:
            print(f"Processed {count} lines, {total_tokens} tokens")

    pos_frequencies = {pos: count / total_tokens for pos, count in pos_counter.items()}

    return pos_frequencies, total_tokens, docs


def plot_pos_distribution(pos_frequencies):
    # Bar plot
    plt.figure(figsize=(10, 5))
    tags = list(pos_frequencies.keys())
    values = list(pos_frequencies.values())

    plt.bar(tags, values)
    plt.title("POS Tag Distribution")
    plt.xlabel("Part of Speech")
    plt.ylabel("Relative Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- RUN ---
file_path = "example.txt"
pos_frequencies, total_tokens, docs = calculate_pos_frequencies(file_path)

print("POS Frequencies:", pos_frequencies)
print("Total Tokens:", total_tokens)

# Plot the distribution
plot_pos_distribution(pos_frequencies)

# Render displacy for the first 1–3 sentences
for doc in docs[:3]:
    displacy.render(doc, style="dep", jupyter=True)  # use style="dep" or "ent"
