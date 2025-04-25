import nltk
from nltk.tokenize import word_tokenize
import string
import re

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def processed_text(text): 
    # Remove numbers
    text_without_numbers = re.sub(r'\d+', '', text)
    
    # Remove URLs
    text_without_urls = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text_without_numbers)
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation.replace("'", ""))
    text_without_punctuation = text_without_urls.translate(translator)
   
    # Convert to lowercase
    final_text_lowercase = text_without_punctuation.lower()
    
    return final_text_lowercase

def calculate_mattr(tokens, window_size=50):
    """
    Compute the Moving-Average Type-Token Ratio (MATTR)
    :param tokens: List of tokens from the processed text
    :param window_size: Size of the sliding window
    :return: The average TTR over all sliding windows
    """
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens)  # Fall back to standard TTR if text is shorter than window
    
    ttr_values = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttr_values.append(len(set(window)) / window_size)
    
    return sum(ttr_values) / len(ttr_values)

def read_file_calculate_mattr(input_text, window_size=50):
    with open(input_text, 'r', encoding='utf-8') as input_file:  # New: Ensure UTF-8 encoding for compatibility
        text = input_file.read()
        
        # Preprocess text
        final_text = processed_text(text)
        
        # Tokenize text
        tokens = word_tokenize(final_text)
        
        if not tokens:  # New: Handle empty files 
            return 0.0
        
        # Calculate MATTR
        return calculate_mattr(tokens, window_size)

# Allow dynamic window size input from the user
window_size = 50  # New: Set default window size
try:
    user_input = input("Enter window size for MATTR calculation (default is 50): ")  # New: User input for customization
    if user_input.strip():
        window_size = int(user_input)
except ValueError:
    print("Invalid input. Using default window size of 50.")  # New: Handle incorrect input gracefully

# Calculate MATTR for both texts
final_mattr_human = read_file_calculate_mattr("human_written.txt", window_size)
print("Human written MATTR:", final_mattr_human)

final_mattr_computer = read_file_calculate_mattr("computer_generated.txt", window_size)
print("Computer generated MATTR:", final_mattr_computer)
