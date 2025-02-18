from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import time
import glob

# Project by Evan Strobel
# 2/18/2025

# POTENTIAL IMPROVEMENTS
# Menu and query entry (DONE)
# List all PDF files in directory (DONE)
#    Allow user to choose which file to read from (DONE)
# Allow user to control LLM parameters (DONE)
# Allow user to choose LLM from list (DONE)
# Change file directory during execution
# Better input validation
# Save file embeddings for reuse
# Better parameter explanations + ranges
# Maybe use ID Mapping to give page numbers of results

# Extracts text from all pages of a PDF split between pages
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():  # Ignore empty pages
            sections.append(text)
    return sections  # List of page-based text chunks

text_chunk_size = 300 # Amount of characters in each text chunk

text_chunk_overlap = 150 # Overlap between text chunks

# Splits the pages of the document into text chunks
def split_text_sections(text_sections, chunk_size=text_chunk_size, overlap=text_chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_sections = text_splitter.split_text("\n\n".join(text_sections))
    return split_sections

# Replaces various bullet point symbols with a consistent character
def normalize_bullets(text):
    return re.sub(r"[\u2022\u25E6\u2043•○▪▶]", "-", text)


directory_path = "data" # Directory where PDFs can be found
filetype = ".pdf" # Filetype to look for in directory
pdf_path = "data/AragornWiki.pdf" # Path of currently selected PDF file
query = "How does Aragorn defeat Sauron?" # Placeholder query


# Converts text to lowercase and removes special characters
# Ensures consistent embeddings for database entries and user query
# (Current unused)
def preprocess_text(text):
    text = text.lower()
    #text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()


# Load a pre-trained embedding model
# Alternative models: all-MiniLM-L6-v2   all-mpnet-base-v2
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Converts text sections into embeddings
def embed_text_sections(text_chunks):
    return np.array(embedding_model.encode(text_chunks)).astype("float32")

topk = 5 # The amount of text sections to retrieve

# Finds the most relevant sections of the PDF based on the query.
def retrieve_relevant_sections(query, top_k=topk):
    query_embedding = np.array([embedding_model.encode(query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)  # Retrieve closest matches
    
    return [extracted_text[i] for i in indices[0]]  # Return the matched sections

# Combines retrieved sections into a single input for summarization
def format_text_for_summary(retrieved_sections, max_words=500):
    combined_text = "\n".join(retrieved_sections)  # Merge retrieved sections
    words = combined_text.split()[:max_words]  # Truncate if too long
    return " ".join(words)  # Return truncated text

# Create the LLM prompt
def CreatePrompt(query, retrieved_text):
    prompt = f"""
    Answer the question with the context given.

    QUESTION: {query}

    CONTEXT: {retrieved_text}
    """



    print("PROMPT: " + prompt + '\n\n')
    return prompt


# Endpoint Query Settings
HF_TOKEN = ""
model_name = "facebook/bart-large-cnn"
llmoptions = ["facebook/bart-large-cnn", "google/flan-t5-large", "HuggingFaceTB/cosmo-1b"]
url = f"https://api-inference.huggingface.co/models/{model_name}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

max_new_tokens = 250
llm_temperature = 0.4
llm_top_p = 0.5
llm_top_k = 40
llm_repetition_penalty = 1.2

def query_llm(prompt, max_retries=3, wait_time=5):
    
    # Queries the LLM endpoint with retry logic in case of errors.
    
    # Args:
    # - prompt (str): The prompt to send to the LLM.
    # - max_retries (int): Maximum number of times to retry the request.
    # - wait_time (int): Time in seconds to wait between retries.

    # Returns:
    # - str: Generated text response or an error message.

    
    payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": max_new_tokens,   # Limit response length (max 250)
        "temperature": llm_temperature,      # Balance between randomness (1) and determinism (0)
        "top_p": llm_top_p,            # Nucleus sampling (lower = more focused)
        "top_k": llm_top_k,             # Limits number of considered tokens (lower = more deterministic)
        "repetition_penalty": llm_repetition_penalty,  # Reduce word repetition (higher = less repetitive)
        }
    }

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()[0]['summary_text']
        
        print(f"Attempt {attempt + 1} failed: {response.status_code}, {response.text}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Wait before retrying
        
    return "Error: LLM request failed after multiple attempts."

editable_parameters = {"text_chunk_size": text_chunk_size, 
                       "text_chunk_overlap": text_chunk_overlap, 
                       "topk": topk, 
                       "max_new_tokens": max_new_tokens, 
                       "llm_temperature": llm_temperature, 
                       "llm_top_p": llm_top_p, 
                       "llm_top_k": llm_top_k, 
                       "llm_repetition_penalty": llm_repetition_penalty}
parameter_keys = list(editable_parameters.keys())

# Program Menu
menuChoice = 9999
while (menuChoice != "0"):
    print(f"""
      Current LLM: {model_name}
      Current File: {pdf_path}

      Select an option:
      1. Query LLM
      2. Change LLM
      3. Change File
      4. Change LLM Parameters
      5. Enter HuggingFace Token (Required for use)
      0. Exit\n""")
    menuChoice = input("Select an option: ")
    if (menuChoice == "1"):
        query = input("Enter query, or type 'CANCEL' to cancel: ")
        if (query != "CANCEL"):
            print("\nThinking...\n")
            extracted_text = extract_text_from_pdf(pdf_path)
            extracted_text = split_text_sections(extracted_text)

            embeddings = embed_text_sections(extracted_text)

            # Use FAISS Indexing for similarity search
            dimension = embeddings.shape[1] 
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            retrieved_sections = retrieve_relevant_sections(query)
            retrieved_text = format_text_for_summary(retrieved_sections)
            retrieved_text = normalize_bullets(retrieved_text)

            response = query_llm(CreatePrompt(query, retrieved_text))
            print("RESPONSE: ", response)
    elif (menuChoice == "2"):
        llmchoice = -1
        while int(llmchoice) < 1 or int(llmchoice) > len(llmoptions):
            print("Model options:")
            for index, item in enumerate(llmoptions, start=1):
                print(f"{index}. {item}")
            llmchoice = input("Select an option: ")
            try:
                llmchoice = int(llmchoice.strip())
                if int(llmchoice) < 1 or int(llmchoice) > len(llmoptions):
                    print("Please enter a valid number.\n")
                else:
                    model_name = llmoptions[llmchoice-1]
            except ValueError:
                print("Please enter a valid number.\n")
    elif (menuChoice == "3"):
        files = glob.glob(f"{directory_path}/*{filetype}")
        filechoice = -1
        while int(filechoice) < 1 or int(filechoice) > len(files): 
            for index, item in enumerate(files, start=1):
                print(f"{index}. {item}")
            filechoice = input("Select a file: ")
            try:
                filechoice = int(filechoice.strip())
                if int(filechoice) < 1 or int(filechoice) > len(files):
                    print("Please enter a valid number.\n")
                else:
                    pdf_path = files[filechoice-1]
            except ValueError:
                print("Please enter a valid number.\n")
    elif (menuChoice == "4"):
        parameterchoice = -1
        while parameterchoice < 1 or parameterchoice > len(editable_parameters):
            for index, (key, value) in enumerate(editable_parameters.items(), start=1):
                print(f"{index}. {key}: {value}")
            parameterchoice = input("Choose a parameter to edit: ")
            try:
                parameterchoice = int(parameterchoice.strip())
            except ValueError:
                print("Please enter a valid number.\n")
            if int(parameterchoice) < 1 or int(parameterchoice) > len(editable_parameters):
                print("Please enter a valid number.\n")
            else:
                selected_key = parameter_keys[parameterchoice-1]
                selected_value = editable_parameters[selected_key]
                newvalue = input(f"Enter new value for {selected_key}: ")
                editable_parameters[selected_key] = newvalue
    elif (menuChoice == "5"):
        HF_TOKEN = input("Enter token: ")