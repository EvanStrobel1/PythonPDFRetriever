# Python LLM PDF Data Retriever 🤖

## Project Overview 📖
Hello! This Python project splits text from a PDF document and utilizes an LLM to summarize. The breakdown of the process is as follows:

1. Takes all the text from a PDF document, split between pages
2. Splits all of the text into smaller text chunks with optional text overlap
3. Uses an embedding model to embed every text chunk
4. Creates a FAISS index map for relevancy searching
5. Takes the user's query and embeds it
6. Searches the FAISS index for the most relevant text sections
7. Presents the LLM with the relevant sections and user's query. The LLM summarizes the sections based on the user's query.

The idea is that an LLM can help a user quickly find relevant information from a large PDF document. 
There are lots of improvements that can be made, but it works pretty well so far!
