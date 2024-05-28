# function to encode text
def text_encoder(original_text_input):
    encoded_text = []
    for char in original_text_input:
        # encoding
        ref_dict = {char: index for index, char in enumerate("AaBegiKIMNnOPpqSsTUWwYy,;:!'")}
        encoding_dict = {char: index for index, char in enumerate("АаВеցіΚӏΜΝոΟΡрԛЅѕΤՍԜԝΥу‚;꞉ǃʾ")}
                 
        invert_ref_dict = {value: key for key, value in ref_dict.items()}
        invert_encoding_dict = {value: key for key, value in encoding_dict.items()}

        if char in ref_dict:
            index = ref_dict[char]
            new_char = invert_encoding_dict[index]
            encoded_text.append(new_char)
        else:
            encoded_text.append(char)

    encoded_string = ''.join(encoded_text)

    # Use a unicode whitespace character to separate each letter
    whitespace_character = "\u200B"  # Four-Per-Em Space character

    final_text = whitespace_character.join(encoded_string)

    encoding_score = 0
    encoded_ch = ""
    for ch in encoded_text:
        if ch in encoding_dict:
            encoding_score += 1
            if ch not in encoded_ch:
                encoded_ch += ch
    
    whitespace_count = 0
    letter_count = 0
    for char in final_text:
        if char == "\u200B":
            whitespace_count += 1
        else:
            letter_count += 1
            
    
    proportion_of_encoding = str(round(encoding_score / len(encoded_text),2))
    proportion_of_whitespace = str(round(whitespace_count/(letter_count-1),2))

    return [final_text, proportion_of_encoding, encoded_ch, proportion_of_whitespace]

