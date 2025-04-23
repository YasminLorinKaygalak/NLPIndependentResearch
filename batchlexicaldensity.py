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

def lexical_density(doc, method=0):
    """
    Calculates the lexical density of a processed spaCy document.
    Lexical Density = (Number of Lexical Items) / (Number of Ranking Clauses)
    Lexical Items include nouns, verbs, adjectives, and adverbs.
    :param doc: The processed spaCy document.
    :param method: The method for calculation (0 = Ure's definition, 1 = Halliday's definition).
    :return: Lexical density score.
    """
    if method == 1:  # Halliday's Definition
        lexical_item_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        lexical_item_count = sum(1 for token in doc if token.pos_ in lexical_item_pos)
        ranking_clause_count = count_ranking_clauses(doc)
        lexical_density_score = lexical_item_count / ranking_clause_count if ranking_clause_count > 0 else 0
    else:  # Ure's Definition
        content_word_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        content_words = [token.text for token in doc if token.pos_ in content_word_pos]
        total_words = [token.text for token in doc if not token.is_punct and not token.is_space]
        lexical_density_score = len(content_words) / len(total_words) if len(total_words) > 0 else 0
    return lexical_density_score

def batch_lexical_density(texts, method=0):
    """
    Processes multiple texts and calculates their lexical densities using spaCy's nlp.pipe for efficiency.
    :param texts: List of input texts to analyze.
    :param method: The method for calculation (0 = Ure's definition, 1 = Halliday's definition).
    :return: List of tuples containing text and its lexical density score.
    """
    nlp = spacy.load("en_core_web_sm")
    results = []
    for doc in nlp.pipe(texts, batch_size=50):
        ld_score = lexical_density(doc, method)
        results.append((doc.text, ld_score))
    return results

 # # Visualize dependency parsing (highlighting ROOTs)
        # print("\nVisualizing Dependency Parsing (Halliday's Definition):")
        # displacy.render(doc, style="dep", jupyter=True)
        # html = displacy.render(doc, style="dep")
        # with open("dependency_visualization.html", "w") as file:
        #     file.write(html)
        
# Example usage
texts = [
    '''In the early days when engineers had to make a bridge across a valley 
       and the valley had a river flowing through it, they often built viaducts, 
       which were constructed of masonry and had numerous arches in them; 
       and many of these viaducts became notable.''',
    '''The quick brown fox jumps over the lazy dog.''',
    '''Artificial intelligence is transforming industries worldwide.'''
]

# Calculate lexical densities for all texts using Ure's definition
results = batch_lexical_density(texts, method=0)
for text, ld in results:
    print(f"Text: {text}\nLexical Density (Ure's Definition): {ld:.4f}\n")

# Calculate lexical densities for all texts using Halliday's definition
results = batch_lexical_density(texts, method=1)
for text, ld in results:
    print(f"Text: {text}\nLexical Density (Halliday's Definition): {ld:.4f}\n")
