import spacy
from typing import List, Dict

def calculate_tree_depth(token):
    """
    Recursively calculate the depth of the dependency tree for a given token.
    """
    if not list(token.children):
        return 1
    else:
        return 1 + max(calculate_tree_depth(child) for child in token.children)

def count_subordinate_clauses(doc):
    """
    Count the number of subordinate clauses in the document.
    """
    return sum(1 for token in doc if token.dep_ in {'advcl', 'ccomp', 'xcomp', 'acl'})

def count_total_clauses(doc):
    """
    Count the total number of clauses in the document.
    """
    return sum(1 for token in doc if token.dep_ in {'ROOT', 'advcl', 'ccomp', 'xcomp', 'acl'})

def count_noun_phrases(doc):
    """
    Count the number of noun phrases in the document.
    """
    return len(list(doc.noun_chunks))


def analyze_syntactic_complexity(texts: List[str]) -> List[Dict[str, int]]:
    """
    Analyze syntactic complexity metrics for a list of texts.
    :param texts: List of input texts to analyze.
    :return: List of dictionaries containing syntactic complexity metrics for each text.
    """
    nlp = spacy.load("en_core_web_sm")
    results = []

    for doc in nlp.pipe(texts, batch_size=50):
        # Calculate depth of the dependency tree
        tree_depth = calculate_tree_depth(next(token for token in doc if token.dep_ == 'ROOT'))

        # Count subordinate clauses
        subordinate_clauses = count_subordinate_clauses(doc)

        # Count total clauses
        total_clauses = count_total_clauses(doc)

        # Count noun phrases
        noun_phrases = count_noun_phrases(doc)

        results.append({
            "Tree Depth": tree_depth,
            "Number of Subordinate Clauses": subordinate_clauses,
            "Total Clauses": total_clauses,
            "Number of Noun Phrases": noun_phrases
        })

    return results

# def visualize_dependency_tree(text: str):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     displacy.render(doc, style="dep", jupyter=True)

# Example usage
if __name__ == "__main__":
    texts = [
        "I know that she will arrive soon because she called me earlier.",
        "Despite the rain, the match continued as scheduled.",
        "The book, which was published last year, has won several awards."
    ]

    complexity_results = analyze_syntactic_complexity(texts)

    for i, text in enumerate(texts):
        print(f"Text: {text}")
        for key, value in complexity_results[i].items():
            print(f"{key}: {value}")
        print("\n")
