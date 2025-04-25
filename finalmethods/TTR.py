import nltk
from nltk.tokenize import word_tokenize
import string
import re

def processed_text(text): 
    # Remove numbers
    text_without_numbers = re.sub(r'\d+', '', text)
    
    # Remove URLs
    text_without_urls = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text_without_numbers)
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation.replace("'",""))
    text_without_punctuation = text_without_urls.translate(translator)
   
    # Convert to lowercase
    final_text_lowercase = text_without_punctuation.lower()
    
    # Return final version of the text
    return final_text_lowercase


def calculate_token_type_ratio(text_without_punctuation):

    # Tokenize the text into words
    tokens = nltk.word_tokenize(text_without_punctuation)

    # Calculate the number of unique tokens
    num_unique_tokens = len(set(tokens))

    # Calculate the total number of tokens
    total_tokens = len(tokens)

    # Calculate the token type ratio
    token_type_ratio = num_unique_tokens / total_tokens

    return token_type_ratio


def read_file_calculate_ttr(input_text):
 
     with open(input_text, 'r') as input_file:
        # Read the content of the input file
        text = input_file.read()
        
        # Convert it to processed version
        final_text = processed_text(text)
        
        # Calculate the ttr for the processed version of the text
        text_token_type_ratio = calculate_token_type_ratio(final_text)
        return text_token_type_ratio
   
final_ttr_human = read_file_calculate_ttr("human_written.txt")
print("Human written: " , final_ttr_human)
final_ttr_computer = read_file_calculate_ttr("computer_generated.txt")
print("Computer generated: ", final_ttr_computer)









