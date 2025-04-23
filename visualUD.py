import spacy
from spacy import displacy

def ud_average_summed_distance_per_sentence(input_text: str, visualize: bool = False) -> float:
    # Load the English spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    with open(input_text, 'r', encoding='utf-8') as input_file:
        text = input_file.read()

    total_summed_distance = 0
    num_sentences = 0

    split_text = text.split("\n")
    all_vis_sentences = []  # for displacy visualization

    for chunk_num in range(100):
        chunk = split_text[chunk_num * 550:(chunk_num + 1) * 550]
        if not chunk:
            continue
        print(f"Processing lines {chunk_num * 550} to {(chunk_num + 1) * 550}")

        doc = nlp("\n".join(chunk))
        print("\tChunk parsed.")

        for sentence in doc.sents:
            sentence_distance = 0
            for token in sentence:
                if token.is_punct or token.dep_ == "ROOT":
                    continue
                distance = abs(token.i - token.head.i)
                sentence_distance += distance

            total_summed_distance += sentence_distance
            num_sentences += 1

            # Store for visualization if requested
            if visualize and len(all_vis_sentences) < 3:
                all_vis_sentences.append(sentence)

    if num_sentences == 0:
        return 0.0

    average_distance_per_sentence = total_summed_distance / num_sentences

    # Visualize a few sentences using displacy
    if visualize and all_vis_sentences:
        print("Launching displacy visualizer in your browser...")
        displacy.serve(all_vis_sentences, style="dep", page=True)

    return average_distance_per_sentence

# Example usage
input_text = 'human_written.txt'
average_dist = ud_average_summed_distance_per_sentence(input_text, visualize=True)
print(f"Average Dependency Distance per Sentence: {average_dist:.4f}")
