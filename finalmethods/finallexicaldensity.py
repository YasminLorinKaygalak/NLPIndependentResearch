import spacy
from collections import Counter
from spacy import displacy


def count_ranking_clauses(doc):
    """
    Estimates the number of ranking clauses in a text.
    A ranking clause typically contains a finite verb.
    """
    finite_verbs = {"VERB"}
    ranking_clause_count = sum(1 for token in doc if token.pos_ in finite_verbs and token.dep_ == "ROOT")

    return ranking_clause_count if ranking_clause_count > 0 else 1  # Avoid division by zero


def lexical_density(text, method=0):
    """
    Calculates the lexical density of a given text using spaCy.
    Lexical Density = (Number of Lexical Items) / (Number of Ranking Clauses)
    Lexical Items include nouns, verbs, adjectives, and adverbs.
    :param text: The input text to analyze.
    :param method: The method for calculation (0 = Ure's definition, 1 = Halliday's definition).
    """
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process text
    doc = nlp(text)

    if method == 1:  # Halliday's Definition
        # Define lexical items (nouns, verbs, adjectives, adverbs)
        lexical_item_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        # Count lexical items
        lexical_item_count = sum(1 for token in doc if token.pos_ in lexical_item_pos)
        # Count ranking clauses
        ranking_clause_count = count_ranking_clauses(doc)
        # Compute lexical density
        lexical_density_score = lexical_item_count / ranking_clause_count if ranking_clause_count > 0 else 0

        print("Lexical Density (Halliday's Definition):", round(lexical_density_score, 4))

        # # Visualize dependency parsing (highlighting ROOTs)
        # print("\nVisualizing Dependency Parsing (Halliday's Definition):")
        # displacy.render(doc, style="dep", jupyter=True)
        # html = displacy.render(doc, style="dep")
        # with open("dependency_visualization.html", "w") as file:
        #     file.write(html)

    else:  # Default to Ure's Definition
        # Define content words (nouns, verbs, adjectives, adverbs)
        content_word_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        # Identify content words and count them
        content_words = [token.text for token in doc if token.pos_ in content_word_pos]
        # Count total words (excluding punctuation)
        total_words = [token.text for token in doc if not token.is_punct and not token.is_space]
        # Compute lexical density
        lexical_density_score = len(content_words) / len(total_words) if len(total_words) > 0 else 0

        print("Content Words Count:", len(content_words))
        print("Total Words Count:", len(total_words))
        print("Content Words:", content_words)
        print("All Words (Excluding Punctuation & Spaces):", total_words)
        print("Lexical Density (Ure's Definition):", round(lexical_density_score, 4))

    # Print dependency info
    for token in doc:
        print(f"{token.text:<15} {token.dep_:<15} {token.head.text:<15} {token.pos_:<10}")

    return lexical_density_score


# Example usage
text = '''In the early days when engineers had to make a bridge across a valley 
          and the valley had a river flowing through it, they often built viaducts, 
          which were constructed of masonry and had numerous arches in them; 
          and many of these viaducts became notable.'''


print("Lexical Density (Ure's Definition):", lexical_density(text))
print("\nLexical Density (Halliday's Definition):", lexical_density(text, method=1))
