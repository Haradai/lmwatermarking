# Function to encode .py files 
import os

def encode_code_file(source_path, dest_path):
    
    try:
        # Extract file extension
        _, file_extension = os.path.splitext(source_path)

        # Initialize a dictionary to map file extensions to comment symbols
        comment_symbols_hash = {
            '.py': '#',
            '.rmd': '#',
            '.php': '#'
            
            # Add more file extensions and their associated comment symbols as needed
        }
        # Initialize list for html and css files to handle separately
        comment_symbols_html_css =[
            '.html',
            '.css'
        ]
            
     
        # Open the source file in read mode
        with open(source_path, 'r', encoding='utf-8') as file:
            # Open the new document in write mode
            with open(dest_path, 'w', encoding='utf-8') as new_file:
                # Initialize variables to store the current line and lines of code
                current_line = ''
                ref_dict = {char: index for index, char in enumerate("AaBegiKIMNnOPpqSsTUWwYy,;:!'")}
                encoding_dict = {char: index for index, char in enumerate("АаВеցіΚӏΜΝոΟΡрԛЅѕΤՍԜԝΥу‚;꞉ǃʾ")}

                # encoding alphabet & punctuation
                invert_ref_dict = {value: key for key, value in ref_dict.items()}
                invert_encoding_dict = {value: key for key, value in encoding_dict.items()}

                # Use a unicode whitespace character to separate each letter
                whitespace_character = "\u200B"  # Four-Per-Em Space character

                '''
                These characters denote comments in most  languages:
                #, //, /*, --, <!--
                e.g Python, Java, Javascript, C, C++, Rust, Go, Typescript, PHP, CSS, R and SQL
                '''
                # Flag to track if we are currently inside a comment block
                inside_comment = False

                if file_extension in comment_symbols_hash: # this is for Python, R or PHP files that exclusively use # for comments
                    for char in file.read():

                        # Check if current character is the start of a comment
                        if char == "#" and not inside_comment:
                            inside_comment = True   # if char is a comment character, flag indicates we are inside a comment block
                            current_line += char
                        
                        # Check if current character is the end of a comment
                        elif char == '\n' and inside_comment:
                            inside_comment = False  # if char is end of line, flag indicates end of comment i.e out of comment block
                            # Write the current line to the new file
                            new_file.write(current_line + '\n')
                            # Reset the current line for the next line of code
                            current_line = ''


                        # Check if the current letter is inside the comment block
                        elif inside_comment:
                            if char in ref_dict:    # If char can be encoded, add the encoded character to current line
                                index = ref_dict[char]
                                current_line += invert_encoding_dict[index]
                                current_line += whitespace_character 
                                
                            elif char == " ":
                                current_line += char
                                current_line += whitespace_character    # add whitespace only at the start of words within comments
                                

                            else:
                                current_line += char # If not, add the current character to the current line
                                current_line += whitespace_character

                        # For all characters that are outside comment blocks entirely
                        elif not inside_comment:
                            current_line += char # Add the current character to the current line


                        # Check if the current character is a newline character
                        if char == '\n' and not inside_comment:   
                            # Write the current line to the new file
                            new_file.write(current_line)
                            # Reset the current line for the next line of code
                            current_line = ''

                elif file_extension in comment_symbols_html_css:
                    inside_HTML_comment = False
                    inside_CSS_comment = False
                    inside_comment = False
                
                    for char in file.read():
                        
                        if char == "/" and not inside_comment: # checking if "//" is present e.g Javascript comment
                            if current_line and current_line[-1] == "/":
                                inside_comment = True   
                            current_line += char
                        
                        elif char == "*" and not inside_comment: # checking if "/*" is present i.e CSS comment
                            if current_line and current_line[-1] == "/":
                                inside_CSS_comment = True   
                            current_line += char
                        
                        elif char == "!" and not inside_HTML_comment: # checking if "<!--" is present i.e HTML comment
                            if current_line and current_line[-1] == "<":
                                inside_HTML_comment = True
                            current_line += char

                        elif char == "D" and inside_HTML_comment:
                            if current_line[-1] == "!" and current_line[-2] == "<": # Ignore <!DOCTYPE> start
                                inside_HTML_comment = False
                            current_line += char

                         # Check if current character is the end of a comment
                        elif char == '\n':
                            inside_comment = False  # if char is end of line, flag indicates end of comment i.e out of comment block
                            # Write the current line to the new file
                            new_file.write(current_line + '\n')
                            # Reset the current line for the next line of code
                            current_line = ''

                        elif char == "/" and inside_comment: # checking if "*/" is present i.e end of CSS comment block "/* ... */"
                            if current_line and current_line[-2] == "*":
                                current_line += char
                                inside_CSS_comment = False  # close flag to indicate end of CSS comment block
                                inside_comment = False
                                # Write the current line to the new file
                                new_file.write(current_line + '\n')
                                # Reset the current line for the next line of code
                                current_line = ''
                        
                        elif char == ">" and inside_HTML_comment: # checking if "->" is present i.e end of HTML comment block "<!--...-->"
                            if current_line and current_line[-1] == "-":
                                current_line += char
                                inside_HTML_comment = False  # close flag to indicate end of CSS comment block
                                inside_comment = False
                                # Write the current line to the new file
                                new_file.write(current_line + '\n')
                                # Reset the current line for the next line of code
                                current_line = ''

                        
                        # Check if the current letter is inside the Javascript comment block
                        elif inside_comment:
                            if char in ref_dict:    # If char can be encoded, add the encoded character to current line
                                index = ref_dict[char]
                                current_line += invert_encoding_dict[index]
                                current_line += whitespace_character 
                                
                            elif char == " ":
                                current_line += char
                                current_line += whitespace_character    # add whitespace only at the start of words within comments
                                
                            else:
                                current_line += char # If not, add the current character to the current line
                                current_line += whitespace_character

                        elif inside_HTML_comment:
                            if char in ref_dict:    # If char can be encoded, add the encoded character to current line
                                index = ref_dict[char]
                                current_line += invert_encoding_dict[index]
                                # no whitespace for HTML comment characters, it is a visible difference 
                                
                            elif char == " ":
                                current_line += char
                                current_line += whitespace_character    # add whitespace only at the start of words within comments
                                
                            else:
                                current_line += char # If not, add the current character to the current line
                                # no whitespace between any letters

                        elif inside_CSS_comment:
                            if char in ref_dict:    # If char can be encoded, add the encoded character to current line
                                index = ref_dict[char]
                                current_line += invert_encoding_dict[index]
                                current_line += whitespace_character
                                 
                                
                            elif char == " ":
                                current_line += char
                                current_line += whitespace_character
                                
                            else:
                                current_line += char # If not, add the current character to the current line
                                current_line += whitespace_character



                        # For all characters that are outside comment blocks entirely
                        elif not inside_comment and not inside_CSS_comment:
                            if not inside_HTML_comment:
                                current_line += char # Add the current character to the current line


                        # Check if the current character is a newline character
                        if char == '\n' and not inside_comment:   
                            # Write the current line to the new file
                            new_file.write(current_line)
                            # Reset the current line for the next line of code
                            current_line = ''
                
                else: # for all files that are not Python, R, PHP, HTML or CSS
                    # Iterate through each character in the file
                    
                    for char in file.read():
                        
                        if char == "/" and not inside_comment: # checking if "//" is present e.g Java comment
                            if current_line and current_line[-1] == "/":
                                inside_comment = True   
                            current_line += char
                        
                        elif char == "-" and not inside_comment: # checking if "--" is present i.e SQL comment
                            if current_line and current_line[-1] == "-":
                                inside_comment = True   
                            current_line += char


                        # Check if current character is the end of a comment
                        elif char == '\n' and inside_comment:
                            inside_comment = False  # if char is end of line, flag indicates end of comment i.e out of comment block
                            # Write the current line to the new file
                            new_file.write(current_line + '\n')
                            # Reset the current line for the next line of code
                            current_line = ''
                                               

                        # Check if the current letter is inside the comment block
                        elif inside_comment:
                            if char in ref_dict:    # If char can be encoded, add the encoded character to current line
                                index = ref_dict[char]
                                current_line += invert_encoding_dict[index]
                                current_line += whitespace_character 
                                
                            elif char == " ":
                                current_line += char
                                current_line += whitespace_character    # add whitespace only at the start of words within comments
                                

                            else:
                                current_line += char # If not, add the current character to the current line
                                current_line += whitespace_character

                        # For all characters that are outside comment blocks entirely
                        elif not inside_comment:
                            current_line += char # Add the current character to the current line


                        # Check if the current character is a newline character
                        if char == '\n' and not inside_comment:   
                            # Write the current line to the new file
                            new_file.write(current_line)
                            # Reset the current line for the next line of code
                            current_line = ''

        print("Contents copied from source file to destination file.")
        return True
    
    except Exception as e:
        print("An error occurred:", str(e))
        return False



