import os
import logging
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from Functions.Encoding.encoding_text import text_encoder
from Functions.Encoding.encoding_text_file import *
from Functions.Encoding.encoding_code_file import encode_code_file
from Functions.Detection.detection_text_file import *
from Functions.Detection.detection_code_file import *
from Functions.Detection.detection_text import *

# Select which html to use
use_form = 'form_v2.html'

# Create flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


###LLMs loading
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList # type: ignore
import torch # type: ignore

device = torch.device("cpu")
llm_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm_model.eval()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # type: ignore
t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
# t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5_txt2txt_checkpoint")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

##Watermark processor and detector
from extended_watermark_processor import WatermarkLogitsProcessor
watermark_processor = WatermarkLogitsProcessor(vocab=list(llm_tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.3,
                                               seeding_scheme="selfhash")

t5_watermark_processor = WatermarkLogitsProcessor(vocab=list(llm_tokenizer.get_vocab().values()),
                                               gamma=0.4,
                                               delta=50,
                                               seeding_scheme="selfhash")


from extended_watermark_processor import WatermarkDetector
watermark_detector = WatermarkDetector(vocab=list(llm_tokenizer.get_vocab().values()),
                                        gamma=0.4, # should match original setting
                                        seeding_scheme="selfhash", # should match original setting
                                        device=llm_model.device, # must match the original rng device type
                                        tokenizer=llm_tokenizer,
                                        z_threshold=2.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)

@app.route('/')
def index():
    return render_template(use_form)

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    add_homoglyphs = False
    
    user_input = request.form['chatbotInput']

    to_chatbot_input = f'''
    <|system|>
    You are a friendly and useful chatbot. Your responses are a paragraph long.</s>
    <|user|>
    {user_input}</s>
    <|assistant|>'''
    
    tokenized_input = llm_tokenizer(to_chatbot_input, return_tensors='pt').to(llm_model.device)

    ##watermarked
    output_tokens = llm_model.generate(**tokenized_input,
                                logits_processor=LogitsProcessorList([watermark_processor]),
                                max_new_tokens=100)


    output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]
    wm_output_text = llm_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    ##unwatermarked
    output_tokens = llm_model.generate(**tokenized_input,max_new_tokens=100)
    output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]
    output_text = llm_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    if add_homoglyphs:
        wm_output_text = text_encoder(wm_output_text)[0]
    
    return render_template('chatbot_response.html', chatbot_response=output_text, chatbot_response_watermarked=wm_output_text)


# Encoding text manually typed in the box
@app.route('/process_homoglyph', methods=['POST'])
def process_homoglyph():
    original_text_input = request.form['encode']
    
    encoded_output = text_encoder(original_text_input)[0]
    proportion_of_encoding = text_encoder(original_text_input)[1]
    encoded_characters = text_encoder(original_text_input)[2]
    whitespace_characters = text_encoder(original_text_input)[3]
    
    # Encoding and returning the details of homoglyphs
    return render_template(use_form, encoded_output=encoded_output, 
                           proportion_of_encoding = proportion_of_encoding,
                           encoded_characters = encoded_characters,
                           whitespace_characters = whitespace_characters)


def t5_generate(model,input_ids:torch.tensor, attention_mask:torch.tensor, watermark_processor=None) -> str:
    model.eval()
    n_tokens_in = torch.sum(attention_mask).item()
    
    decoder_input_ids = torch.tensor([[0,0,0,0,0]]).to(model.device)
    for _ in range(n_tokens_in):

        #with and without watermark
        if watermark_processor:
            out = model.generate(input_ids = input_ids,attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, 
                                max_new_tokens=1,
                                logits_processor=LogitsProcessorList([t5_watermark_processor]))
        else:
            out = model.generate(input_ids = input_ids,attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, 
                                max_new_tokens=1)
        
        decoder_input_ids = out
    
    #return new text
    return t5_tokenizer.decode(decoder_input_ids[0])

# Encoding text manually typed in the box
@app.route('/process_t5', methods=['POST'])
def process_t5():
    original_text_input = request.form['encode']
    
    tokenized_input = t5_tokenizer(original_text_input, return_tensors='pt').to(t5_model.device)

    ##watermarked
    output_text = t5_generate(t5_model,tokenized_input["input_ids"],tokenized_input["attention_mask"], watermark_processor)
    
    # Encoding and returning the details of homoglyphs
    return render_template(use_form, encoded_output=output_text)

# Encoding text file
@app.route('/upload_text', methods=['POST'])
def upload_text_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file to a temporary location
        input_file_path = secure_filename(file.filename)
        file.save(input_file_path)

        # Get the suffix of the code file
        file_extension = os.path.splitext(input_file_path)[1]

        # Define the output file path
        output_file_path = "C:/Users/Andrew/OneDrive - Singapore Management University/SMU stuff/3.2 Exchange/Social Innovation/Creating-watermark-for-text/Functions/Output/Encoded text"+file_extension   # Replace with the path to the output Word file
        
        # Modify the Word document
        paragraph_list = read_words_from_word_file_with_paragraphs(input_file_path)
        
        # Write words to the output file 
        write_words_to_word_file_with_paragraphs(paragraph_list, output_file_path)

        # Send the modified document as a file download
        return send_file(output_file_path, as_attachment=True)
    
# Encoding code file
@app.route('/upload_code', methods=['POST'])
def upload_code_file():
    if 'code_file' not in request.files:
        return "No file part"
    
    file = request.files['code_file']

    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file to a temporary location
        input_file_path = secure_filename(file.filename)
        file.save(input_file_path)

        # Get the suffix of the code file
        file_extension = os.path.splitext(input_file_path)[1]

        # Define the output file path
        output_file_path = "C:/Users/Andrew/OneDrive - Singapore Management University/SMU stuff/3.2 Exchange/Social Innovation/Creating-watermark-for-text/Functions/Output/Encoded code"+file_extension   # Replace with the path to the output Word file
             
        # Write code to the output file 
        encode_code_file(input_file_path, output_file_path)
        print(output_file_path)

        # Send the modified document as a file download
        return send_file(output_file_path, as_attachment=True)

# Detecting text manually typed in the box
@app.route('/detect_text', methods=['POST'])
def detect_text():
    remove_homoglyphs = False

    encoded_text_input = request.form['detect']

    text_input_with_flags = flagged_string_with_html(encoded_text_input)
    
    homoglyph_proportion = homoglyph_detection(encoded_text_input)[0]
    whitespace_proportion = homoglyph_detection(encoded_text_input)[1]
    homoglyph_list = homoglyph_detection(encoded_text_input)[2]
    
    # After they have been detected, remove the homoglyphs for the other watermark detector
    homoglyphs = list("АаВеցіΚӏΜΝոΟΡрԛЅѕΤՍԜԝΥуΖ‚;꞉ǃʾ")
    normal_characters = list("AaBegiKIMNnOPpqSsTuWwYyZ,;:!")
    homo2normal = {homo:normal for homo, normal in zip(homoglyphs, normal_characters)}

    if remove_homoglyphs:
        homo_text = list(encoded_text_input)
        for i in range(len(homo_text)):
            if homo_text[i] in list(homo2normal.keys()):
                homo_text[i] = homo2normal[homo_text[i]]

        encoded_text_input = ''.join(homo_text)   

    score_dict = watermark_detector.detect(encoded_text_input) # or any other text of interest to analyze
    
    if score_dict['prediction'] == False:
        score_dict['confidence'] = 0

    # Detecting and returning the details of homoglyphs
    return render_template(use_form, text_input_with_flags = text_input_with_flags,
                           homoglyph_proportion = homoglyph_proportion, 
                           whitespace_proportion = whitespace_proportion,
                           homoglyph_list = homoglyph_list,
                           num_tokens = score_dict["num_tokens_scored"],
                           num_green_tokens = score_dict["num_green_tokens"],
                           green_fraction = round(score_dict["green_fraction"],2),
                           z_score = round(score_dict["z_score"],2),
                           prediction = score_dict["prediction"],
                           confidence = score_dict["confidence"]
                           ) 

# Detecting watermarks in text file uploaded
@app.route('/detect_textfile', methods=['POST'])
def detect_text_file():
    
    if 'text_file_detect' not in request.files:
        return "No file part"
    
    file = request.files['text_file_detect']
    

    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file to a temporary location
        input_file_path = secure_filename(file.filename)
        file.save(input_file_path)

        # Get the suffix of the code file
        file_extension = os.path.splitext(input_file_path)[1]

        # Define the output file path
        output_file_path = "C:/Users/Andrew/OneDrive - Singapore Management University/SMU stuff/3.2 Exchange/Social Innovation/Creating-watermark-for-text/Functions/Output/Evaluated text"+file_extension   # Replace with the path to the output Word file

        proportion_of_homoglyphs = read_encoded_characters_from_word_file_with_paragraphs(input_file_path,output_file_path)[0]
        proportion_of_whitespaces = read_encoded_characters_from_word_file_with_paragraphs(input_file_path,output_file_path)[1]
    
    # Encoding and returning the details of homoglyphs
    return render_template(use_form, proportion_of_homoglyphs = proportion_of_homoglyphs,
                           proportion_of_whitespaces = proportion_of_whitespaces)

# Detecting watermarks in code file uploaded
@app.route('/detect_codefile', methods=['POST'])
def detect_code_file():
    
    if 'code_file_detect' not in request.files:
        return "No file part"
    
    file = request.files['code_file_detect']
    

    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file to a temporary location
        input_file_path = secure_filename(file.filename)
        file.save(input_file_path)

       

        # Define the output file path
        output_file_path = "C:/Users/Andrew/OneDrive - Singapore Management University/SMU stuff/3.2 Exchange/Social Innovation/Creating-watermark-for-text/Functions/Output/Evaluated code.doc"   # Replace with the path to the output Word file


        read_encoded_characters_from_code_file(input_file_path, output_file_path)
        
    
    # Encoding and returning the details of homoglyphs
    return render_template(use_form)


if __name__ == '__main__':
    app.run(debug=True)
