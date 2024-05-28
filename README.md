# Creating watermark for AI-generated text
For effective watermarks within AI-generated text, several key features are essential: they should be hard to remove, easily detectable and non-disruptive to the content's readability and usability (i.e code snippets).

These Python functions take in a text string and encodes the string via changing letters and punctuation marks. They are designed as part of a Python library for Generative AI platforms such as Chat GPT, Copilot, Bard etc., which rely on Large Language Models, can watermark their generated text. When generated texted is copied wholesale from these platforms, a detector can detect the unusual letters and punctuation used in the text and thus determine how much of a written piece is AI-generated.

A total of 28 characters will be substituted with new homoglyphs (Refer to Encoding Matrix to see the full list). These homoglyph substitutes are letters and punctuation marks from other languages (e.g Cyrillic, Armenian, Greek, etc.) and appear virtually identical to their English counterparts. All homoglyphs use different ASCII keys, which will be flagged up instantly by a detector. Each homoglyph can only be written using Unicode, ALT keystroke or a non-English keyboard, thus authors of written work in English cannot justify their use in articles as "accidental" typing.

For AI-generated text, whitespaces with zero-spaces are included either between each character or after every word. This acts an additional layer of watermarking, to combat against human efforts to substitute commonly-used words in their AI-generated writing with synonyms.

In the case of AI-generated code snippets, only code comments can be encoded as substituting characters in code itself would render the code unusable.


The Python functions are as follows:
- Encoding text directly: For Generative AI platforms which generate text word by word from their LLMs (e.g ChatGPT, Copilot, etc), this function is intended to encode each character as it is being generated
- Encoding text files: For platforms which generate full word documents (e.g PDF, .docx, .ppt), this function is intended to read documents in full, encode accordingly, then write to a new file with the encoded text. The function takes in .doc and .txt files only
- **Encoding code files: For platforms which generate code files (e.g .py, .html, .sql, .js, etc), this function encodes all code comments before writing to a new file of usable code. Function is successful for code files excluding .css files and .html containing CSS comments
- Detection for text files: Uploading a text file, it will detect the proportion of homoglyphs and zero-width whitespaces encoded with the text. Function takes in .doc and .txt files only
- Detection for code files: Uploading a code file, it will detect the proportion of homoglyphs and zero-width whitespaces encoded with the code comments.

**Work in Progress


