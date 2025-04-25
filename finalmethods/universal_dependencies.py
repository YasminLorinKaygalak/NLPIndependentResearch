import spacy

def ud_average_summed_distance_per_sentence(input_text: str) -> float:

    # Load the English spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    with open(input_text, 'r') as input_file:
        text = input_file.read()

    # Parse the text using spaCy to obtain the dependencies
    total_summed_distance = 0
    num_sentences = 0
    
    split_text = text.split("\n")
    for chunk_num in range(100):
        chunk = split_text[(chunk_num)*550:(chunk_num+1)*550]
        print((chunk_num)*550,(chunk_num+1)*550)
        
        doc = nlp("\n".join(chunk))
        print("\t", "progress")
        #doc = nlp(text)

        for sentence in doc.sents:
            sentence_distance = 0

            for token in sentence:
                # Skip punctuation tokens and the root nodes
                if token.is_punct or token.dep_ == "ROOT":
                    continue

                # Compute the distance between the token and its head
                distance = abs(token.i - token.head.i)
                sentence_distance += distance

            total_summed_distance += sentence_distance
            num_sentences += 1

    # To avoid dividing by zero
    if num_sentences == 0:
        return 0

    # Calculate the average summed distance per sentence
    average_distance_per_sentence = total_summed_distance / num_sentences

    return average_distance_per_sentence

# Example usage
input_text = 'human_written.txt'
print(ud_average_summed_distance_per_sentence(input_text))

