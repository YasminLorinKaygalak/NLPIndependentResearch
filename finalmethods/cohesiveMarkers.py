import nltk
import string

def count_cohesive_markers(input_text):
    
    # List of cohesive markers of 3 different types
    single_word_markers = [
        "also", "and", "but", "because", "nor", "or", 
        "then", "yet", "first", "firstly", "second", "secondly", "third", "thirdly", "fourth", "fourthly", "next", 
        "finally" , "lastly", "last" , "additionally", "again", "likewise", "similarly", "furthermore", "moreover", "besides", 
        "still", "nevertheless", "nonetheless", "altogether" , "therefore" , "thus",  "accordingly", "consequently", "hence", 
        "now", "so", "anyhow", "anyway", "anyways", "besides" , "else" , "however" , "nevertheless", "nonetheless", "though",
        "conversely", "incidentally", "meantime", "while" , "eventually", "originally", "otherwise", "rather", "somehow", "subsequently",
        "besides", "despite", "except", "like", "too", "unlike"
    ]
   
    multi_word_markers = [
        "in the first place", "in the second place", "in the third place", "first of all", "second of all", 
        "third of all", "for one thing", "to being with", "to start with", "in conclusion" , "to conclude",  "at last",
        "above all" , "once again", "in addition", "in the same way", "by the same token" , "even worse"
        "then again",  "distinguished from" ,  "all in all",  "in sum", "to summarize", "to sum up"
        "as a result", "as in consequence", "as a consequence", "after all", "all the same",
        "at any rate", "at the same time" , "in any case", "in any event", "for all that" ,
        "on the other hand", "better still", "and still", "that said","but then", "but yet",
        "as a matter of fact", "by the way", "in contrast", "in fact", "in the meantime", 
        "on the contrary", "as well", "beause of", "for that reason", "in contrast to", "in contrast with", "in spite of", "instead of",
        "in place of", "in that case", "in the event of", "in this way", "in that way", "and then"
    ]

    paired_markers = {
        "both": "and",
        "either": "or",
        "neither": "nor",
        "not only": "but also"
    }
    
    # Initialize a counter for cohesive markers
    count = 0
    
    # Read the file
    with open(input_text, 'r') as input_file:
        # Normalize the text to lowercase and tokenize by space   
        file = input_file.read().lower()
        sentences = nltk.sent_tokenize(file)
        
        for sentence in sentences:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            tokens = nltk.word_tokenize(sentence)
            # Initialize a list to track processed tokens
            processed = [False] * len(tokens)  

            # Loop through the tokens looking for all the types of cohesive markers
            i = 0
            while i < len(tokens):
                token = tokens[i]
                 
                marker_found = False
                # Check for paired markers
                for start, end in paired_markers.items():
                     # Split the start marker into individual words
                    start_tokens = start.split() 
                    # Split the end marker into individual words
                    end_tokens = end.split()  

                    if tokens[i:i+len(start_tokens)] == start_tokens:
                        found_end = False
                        for j in range(i + len(start_tokens), len(tokens) - len(end_tokens) + 1):
                            if tokens[j:j+len(end_tokens)] == end_tokens:
                                skipped = 0
                                found_end = True
                                break
                            elif len(end_tokens) > 1:
                                #print(tokens[j: j + 1] + tokens[j+2:j+1+len(end_tokens)])
                                if tokens[j: j + 1] + tokens[j+2:j+1+len(end_tokens)] == end_tokens:
                                    skipped = 1
                                    found_end = True
                                    break
                            
                                #print("\t", tokens[j: j + 1] + tokens[j+3:j+2+len(end_tokens)])
                                if tokens[j: j+1] + tokens[j+3:j+2+len(end_tokens)] == end_tokens:
                                    skipped = 2
                                    found_end = True
                                    break
                            
                        if found_end:
                            #print(f"{start} ... {end}")
                            count += 1
                            # Mark all tokens from the start to the end of the paired marker as processed
                            for k in range(i, j + skipped + len(end_tokens)):
                                processed[k] = True
                            i = j + skipped + len(end_tokens) - 1
                            marker_found = True        
                if marker_found:
                    continue
                
                # Check for regular multi-word markers with allowance for words in between
                for marker in multi_word_markers:
                    marker_tokens = marker.split()
                    marker_index = 0  # To keep track of the current position in the marker
                    extra_words_allowed = 1  # Number of extra words allowed between marker tokens
                    found_marker = False

                    for j in range(i, len(tokens)):
                        # Check if the current token matches the current marker token
                        if tokens[j] == marker_tokens[marker_index]:
                            marker_index += 1
                            if marker_index == len(marker_tokens):
                                # Entire marker found
                                found_marker = True
                                break
                        elif marker_index > 0 and extra_words_allowed > 0:
                            # Allow for a certain number of extra words between marker tokens
                            extra_words_allowed -= 1
                        else:
                            # Reset if sequence does not match
                            marker_index = 0
                            extra_words_allowed = 1
                            break

                    if found_marker:
                        count += 1
                        # Mark all tokens from the start to the end of the found marker as processed
                        for k in range(i, j + 1):
                            processed[k] = True
                        i = j  # Update the index to the end of the found marker
                        marker_found = True
                        break
                if marker_found:
                    continue
            
                if not processed[i] and token in single_word_markers:
                    count += 1
                    processed[i] = True
                i += 1

                    
    return count


input_text = 'human_written.txt'
input_text2 = 'computer_generated.txt'
print(count_cohesive_markers(input_text))
print(count_cohesive_markers(input_text2))

