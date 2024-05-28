from docx import Document
import os

def read_words_from_word_file_with_paragraphs(file_path):
    # Extract file extension
    _, file_extension = os.path.splitext(file_path)

    # Initialize an empty list to store the paragraphs
    paragraphs = []
 
    try:
        # Open the Word document
        doc = Document(file_path)

        # Iterate through each paragraph in the document
        for paragraph in doc.paragraphs:
            # Add the original paragraph text to the list
            paragraphs.append(paragraph.text)

        
    except Exception as e:
        # Handle any exceptions that occur during file processing
        print(f"Error reading file: {e}")
    
    return paragraphs

def write_words_to_word_file_with_paragraphs(paragraphs, output_file_path):
    try:
        # Create a new Word document
        doc = Document()

        

        # Add each paragraph to the document
        for paragraph_string in paragraphs:
            # split each paragraph string into a list of word strings
            paragraph_list_of_words = paragraph_string.split()

            # new list of encoded words per paragraph list
            new_paragraph_list_of_words = []

            # iterate through each list of word strings to check each character
            for i in range(len(paragraph_list_of_words)):
                word = paragraph_list_of_words[i]

                new_word = []
                # encoding alphabet & punctuation
                ref_dict = {char: index for index, char in enumerate("AaBegiKIMNnOPpqSsTUWwYy,;:!'")}
                encoding_dict = {char: index for index, char in enumerate("АаВеցіΚӏΜΝոΟΡрԛЅѕΤՍԜԝΥу‚;꞉ǃʾ")}
                 
                invert_ref_dict = {value: key for key, value in ref_dict.items()}
                invert_encoding_dict = {value: key for key, value in encoding_dict.items()}
                
                # Use a unicode whitespace character to separate each letter
                whitespace_character = "\u200B"  # Four-Per-Em Space character

                for char in word:
                    if char in ref_dict:
                        index = ref_dict[char]
                        new_word.append(invert_encoding_dict[index])
    
                    else:
                        new_word.append(char)
                        
                
                new_word_string = ''.join(new_word)

                
                if i == len(paragraph_list_of_words) - 1:
                    new_paragraph_list_of_words.append(new_word_string)
                    new_paragraph_list_of_words.append(whitespace_character) # invisible whitespace is added after every word

                else:
                    new_paragraph_list_of_words.append(new_word_string + " ") # every word that is not last must have whitespace
                    new_paragraph_list_of_words.append(whitespace_character)  # invisible white space is added after the space of every word      

            new_paragraph_string = ''.join(new_paragraph_list_of_words)
            
            doc.add_paragraph(new_paragraph_string)

        # Save the document
        doc.save(output_file_path)

        print(f"Paragraphs written to {output_file_path} successfully.")
    except Exception as e:
        # Handle any exceptions that occur during file writing
        print(f"Error writing file: {e}")

       


