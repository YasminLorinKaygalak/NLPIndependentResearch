import spacy
from spacy import displacy
from typing import List, Dict, Tuple


def calculate_tree_depth(token):
    if not list(token.children):
        return 1
    else:
        return 1 + max(calculate_tree_depth(child) for child in token.children)


def count_subordinate_clauses(doc):
    subordinate_clauses = 0
    for token in doc:
        if token.dep_ in {'advcl', 'ccomp', 'xcomp', 'acl'}:  # Common dependency labels for subordination
            subordinate_clauses += 1
    return subordinate_clauses


def count_total_clauses(doc):
    return sum(1 for token in doc if token.dep_ in {'ROOT', 'advcl', 'ccomp', 'xcomp', 'acl'})


def count_noun_phrases(doc):
    return len(list(doc.noun_chunks))


def analyze_syntactic_complexity(text: str) -> Dict[str, int]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Find the root of the dependency tree
    root = [token for token in doc if token.head == token][0]

    # Calculate depth of the dependency tree
    tree_depth = calculate_tree_depth(root)

    # Count subordinate clauses
    subordinate_clauses = count_subordinate_clauses(doc)

    # Count total clauses
    total_clauses = count_total_clauses(doc)

    # Count noun phrases
    noun_phrases = count_noun_phrases(doc)

    return {
        "Tree Depth": tree_depth,
        "Number of Subordinate Clauses": subordinate_clauses,
        "Total Clauses": total_clauses,
        "Number of Noun Phrases": noun_phrases
    }


# def visualize_dependency_tree(text: str):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     displacy.render(doc, style="dep", jupyter=True)


def batch_analyze(texts: List[str]) -> Dict[str, Tuple[float, float]]:
    results = [analyze_syntactic_complexity(text) for text in texts]

    total_depth = sum(result['Tree Depth'] for result in results)
    total_subordinate_clauses = sum(result['Number of Subordinate Clauses'] for result in results)
    total_clauses = sum(result['Total Clauses'] for result in results)
    total_noun_phrases = sum(result['Number of Noun Phrases'] for result in results)

    avg_depth = total_depth / len(results)
    avg_subordinate_clauses = total_subordinate_clauses / len(results)
    avg_total_clauses = total_clauses / len(results)
    avg_noun_phrases = total_noun_phrases / len(results)

    return {
        'Average Tree Depth': avg_depth,
        'Average Number of Subordinate Clauses': avg_subordinate_clauses,
        'Average Total Clauses': avg_total_clauses,
        'Average Noun Phrases': avg_noun_phrases
    }


def main():
    # Example text for visualization
    text = "I know that she will arrive soon because she called me earlier."
    # visualize_dependency_tree(text)

    # Analyzing single sentence complexity
    result = analyze_syntactic_complexity(text)
    print(f"\nSyntactic Complexity Analysis:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Allowing batch analysis
    texts = [text]
    stats = batch_analyze(texts)
    print(f"\nAverage Syntactic Complexity Across Texts:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
