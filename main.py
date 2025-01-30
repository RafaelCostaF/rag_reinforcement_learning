from Functions.documents import load_documents
from Functions.text import clear_text, remove_stopwords, split_text_into_chunks, remove_portuguese_accents



DOCUMENT_PATHS = ["./Documents/a (1).pdf", "./Documents/FAQ (1).pdf"]
CHUNK_SIZE = 500



# BEGIN = Processing documents 
texts = load_documents(DOCUMENT_PATHS)
cleaned_texts = [clear_text(text) for text in texts]
no_stopwords_text = [remove_stopwords(text) for text in cleaned_texts]
no_accents_portuguese = [remove_portuguese_accents(text) for text in no_stopwords_text]
joined_texts = " ".join(no_accents_portuguese)
joined_texts = joined_texts.lower()
text_chunks = split_text_into_chunks(joined_texts, chunk_size=CHUNK_SIZE)
# END = Processing documents 

print(text_chunks)

