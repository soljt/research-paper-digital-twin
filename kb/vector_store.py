import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import get_titles

from langchain_community.vectorstores import FAISS
import re

from_pdf = False # knowledge base from PDFs (without summarization by GPT) was not very effective - papers are too long

# Try to remove extra text from parsed PDF
def clean_text(text):
    # Remove common sections such as references, authors, institutions, etc.
    
    # Remove preamble (authors, institutions, footnotes)
    text = re.sub(r'(\bAuthors\b.*?(\n|$))', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'(\bAffiliations\b.*?(\n|$))', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove references section
    text = re.sub(r'(\bReferences\b.*)', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove anything related to keywords
    text = re.sub(r'(\bKeywords\b.*?(\n|$))', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove URLs or emails
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    
    # Remove footnotes or small print typically denoted by numbers or special characters
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove additional unwanted patterns, like section headers (optional)
    text = re.sub(r'\bAbstract\b.*?(\n|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\bIntroduction\b', '', text, flags=re.IGNORECASE)
    return text.strip()

# Setup OpenAI API (or you can use any other LLM provider)
openai_api_key = os.environ.get('OPENAI_API_KEY')
print(f"OpenAI API Key: {openai_api_key}")

if from_pdf:
    folder = 'kb/pdfs'
else:
    folder = 'kb/paper_processing_output'

# Load all documents
documents = []
# loader = PyPDFLoader(os.path.join(pdf_folder, 'ssrn-4266038.pdf'))
# docs_lazy = loader.lazy_load()
# for doc in docs_lazy:
#     cleaned_text = clean_text(doc.page_content)
#     if cleaned_text:
#         doc.page_content = cleaned_text
#         documents.append(doc)
for file in os.listdir(folder):
    print(file)
    if (from_pdf and file.endswith('.pdf')):
        loader = PyPDFLoader(os.path.join(folder, file))
        docs_lazy = loader.lazy_load()
        for doc in docs_lazy:
            cleaned_text = clean_text(doc.page_content)
            if cleaned_text:
                doc.page_content = cleaned_text
                documents.append(doc)
    elif (not from_pdf and file.endswith('.txt')):
        loader = TextLoader(os.path.join(folder, file), encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            ssrn_id = file.split("-")[1].split("_")[0]
            doc.metadata['title'] = get_titles.get_paper_title_with_rate_limit(ssrn_id)
            documents.append(doc)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter( # you can alter these parameters as you see fit
    chunk_size=800,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)

############################################################################################################
# Embed the chunks and store them in a database alongside computed embeddings
print("Creating the database...")
try:
    # db = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=openai_api_key)) # use OpenAI embeddings
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    # Parameters for HF embeddings
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}

    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = FAISS.from_documents(docs, hf)
    print("Database created successfully.")
except Exception as e:
    print(f"Error creating FAISS database: {e}")

# Save the database
if from_pdf:
    output_file = "kb/faiss_index_pdf"
else:
    output_file = "kb/faiss_index_hf2"
db.save_local(output_file)



############################################################################################################
# Test functionality
if __name__ == "__main__":
    # print("got here")
    query = "what is the trolley problem?"
    docs = db.similarity_search(query, k=3)
    for idx, doc in enumerate(docs):
        print(f"{idx+1}: \n{doc.page_content}\nMETADATA: {doc.metadata}")