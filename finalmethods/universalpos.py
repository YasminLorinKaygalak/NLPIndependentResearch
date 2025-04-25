import spacy
from collections import Counter

def calculate_pos_frequencies(texts):
    nlp = spacy.load("en_core_web_sm")

    # Counter for POS tags
    pos_counter = Counter()

    # Total number of tokens across all texts
    total_tokens = 0
    
    count = 0
    
    with open(texts, 'r') as input_file:
        text = input_file.read()

    lines = text.split('\n')
    
    # Process each text
    for line in lines:
        count += 1
        if count % 100 == 0:
            print(total_tokens)
            
        # Use spaCy to tokenize the text and extract POS
        doc = nlp(line)
        # Update POS counts and total tokens
        pos_counter.update([token.pos_ for token in doc])
        total_tokens += len(doc)

    # Calculate the relative frequencies
    pos_frequencies = {pos: count / total_tokens for pos, count in pos_counter.items()}

    return pos_frequencies, total_tokens

text = "example.txt"

pos_frequencies, total_tokens = calculate_pos_frequencies(text)
print("POS Frequencies:", pos_frequencies)
print("Total Tokens:", total_tokens)

#If wanted explanation
for pos, freq in pos_frequencies.items():
   explanation = spacy.explain(pos)
   print(f"{pos}: {explanation} - Frequency: {freq}")