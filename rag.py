import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mistralai import Mistral

def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def generate_embeddings(documents, model):
    embeddings = model.encode(documents, convert_to_numpy=True)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    return faiss_index

def search_faiss_index(query_embedding, faiss_index, top_k=3):
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    return indices[0], distances[0]

def rewrite_query_with_mistral(query, api_key):
    mistral = Mistral(api_key)
    
    prompt = f"""Rewrite the following query to make it clearer and more specific
    
    the rewritten query should:
    - PRESERVE the original query intent
    - DONT INTRODUCE new topics or queries different from the original in any way, shape of form
    - DONT EVER ANSWER the original question, instead focus on rephrasing and expanding into a new query
    - KEEP the language of the original query

    Original query: {query}"""
    
    try:
        response = mistral.chat.complete(
            model="mistral-large-latest",
            messages= [
                {
                    "role":"user",
                    "content":prompt,
                },
            ],
            max_tokens=200,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return query

def generate_response_from_mistral(query, context, api_key):
    mistral = Mistral(api_key)
    
    prompt = f"""You are a consultant that should help with extracting information from the text. 
    You can ONLY use information given in the text and NO other external sources.
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Query: {query}
    Answer:"""
    
    try:
        response = mistral.chat.complete(
            model= "mistral-large-latest",
            messages = [
                {
                    "role":"user",
                    "content":prompt,
                },
            ],
            max_tokens=200,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error generating response"

def run_rag_pipeline(query, knowledge_base_file, sentence_model, api_key):
    knowledge_base = load_knowledge_base(knowledge_base_file)

    embeddings = generate_embeddings(knowledge_base, sentence_model)
    
    faiss_index = create_faiss_index(embeddings)
    
    query_embedding = sentence_model.encode([query], convert_to_numpy=True)[0]
    
    top_indices, _ = search_faiss_index(query_embedding, faiss_index)
    
    relevant_context = "\n".join([knowledge_base[idx].strip() for idx in top_indices])
    
    rewritten_query = rewrite_query_with_mistral(query, api_key)
    
    response = generate_response_from_mistral(rewritten_query, relevant_context, api_key)
    
    return response

if __name__ == "__main__":
    api_key = "" # Here goes your mistral API key

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    knowledge_base_file = "knowledge_base.txt"
    
    print("Welcome to the RAG pipeline. Type 'exit' to exit.")
    
    while True:
        query = input("Ask a question (type 'exit' to exit): ")
        if query.lower() == 'exit':
            break
        
        response = run_rag_pipeline(query, knowledge_base_file, sentence_model, api_key)
        print("\nResponse:", response)
