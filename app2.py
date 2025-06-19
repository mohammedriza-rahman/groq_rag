# import streamlit as st
# import os
# from typing import List, Tuple
# import PyPDF2
# from io import BytesIO
# import faiss
# import numpy as np
# from groq import Groq
# from cohere import Client
# import pickle
# import tiktoken

# # Initialize API clients with Streamlit secrets
# groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# cohere_client = Client(api_key=st.secrets["COHERE_API_KEY"])


# def count_tokens(text: str) -> int:
#     encoding = tiktoken.get_encoding("cl100k_base")
#     return len(encoding.encode(text))

# class DocumentProcessor:
#     def __init__(self, chunk_size: int = 500):
#         self.chunks = []
#         self.chunk_size = chunk_size

#     def process_pdf(self, pdf_file: BytesIO) -> List[str]:
#         reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()

#         words = text.split()
#         chunks = []
#         current_chunk = []
#         current_size = 0

#         for word in words:
#             current_chunk.append(word)
#             current_size = count_tokens(" ".join(current_chunk))

#             if current_size >= self.chunk_size:
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk = current_chunk[-50:]  # overlap
#                 current_size = count_tokens(" ".join(current_chunk))

#         if current_chunk:
#             chunks.append(" ".join(current_chunk))

#         self.chunks = chunks
#         return chunks

# class VectorStore:
#     def __init__(self):
#         self.index = None
#         self.texts = []

#     def create_index(self, texts: List[str]):
#         embeddings = cohere_client.embed(
#             texts=texts,
#             model='embed-english-v3.0',
#             input_type='search_query'
#         ).embeddings

#         dimension = len(embeddings[0])
#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(np.array(embeddings).astype('float32'))
#         self.texts = texts

#     def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
#         query_embedding = cohere_client.embed(
#             texts=[query],
#             model='embed-english-v3.0',
#             input_type='search_query'
#         ).embeddings[0]

#         D, I = self.index.search(
#             np.array([query_embedding]).astype('float32'), k
#         )

#         results = [(self.texts[i], D[0][idx]) for idx, i in enumerate(I[0])]
#         return results

# def generate_response(query: str, context: str, max_tokens: int = 4000) -> str:
#     base_prompt = """You are a helpful assistant. Use the following context to answer the question. 
#     If you cannot answer based on the context, say you don't know.

#     Context: {context}

#     Question: {query}

#     Answer:"""

#     prompt_tokens = count_tokens(base_prompt.format(context="", query=query))
#     available_tokens = max_tokens - prompt_tokens - 1024

#     context_tokens = count_tokens(context)
#     if context_tokens > available_tokens:
#         encoding = tiktoken.get_encoding("cl100k_base")
#         context_ids = encoding.encode(context)
#         truncated_ids = context_ids[:available_tokens]
#         context = encoding.decode(truncated_ids)

#     prompt = base_prompt.format(context=context, query=query)

#     try:
#         completion = groq_client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="gemma2-9b-it",
#             temperature=0.3,
#             max_tokens=1024
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# # --- Streamlit UI ---

# st.set_page_config(page_title="📚 PDF Chat Assistant", page_icon="🤖")
# st.title("🤖 Chat with Your Document")

# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = VectorStore()
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# uploaded_file = st.file_uploader("📄 Upload a PDF file to chat with", type="pdf")

# if uploaded_file:
#     with st.spinner("Processing document..."):
#         try:
#             processor = DocumentProcessor(chunk_size=500)
#             chunks = processor.process_pdf(uploaded_file)
#             st.session_state.vector_store.create_index(chunks)
#             st.success("✅ Document processed successfully! Ask anything about it below.")
#         except Exception as e:
#             st.error(f"Document processing failed: {str(e)}")

# # Display past messages like ChatGPT
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if query := st.chat_input("Ask something about your uploaded PDF..."):
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             if st.session_state.vector_store.index is not None:
#                 try:
#                     results = st.session_state.vector_store.search(query, k=2)
#                     context = "\n".join([chunk for chunk, _ in results])
#                     response = generate_response(query, context)
#                 except Exception as e:
#                     response = f"Error: {str(e)}"
#             else:
#                 response = "❗ Please upload a document first!"

#         st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})


import streamlit as st
import os
from typing import List, Tuple
import PyPDF2
from io import BytesIO
import faiss
import numpy as np
from groq import Groq
from cohere import Client
import pickle
import tiktoken

# Initialize API clients with Streamlit secrets
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
cohere_client = Client(api_key=st.secrets["COHERE_API_KEY"])

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500):
        self.chunks = []
        self.chunk_size = chunk_size

    def process_pdf(self, pdf_file: BytesIO) -> List[str]:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size = count_tokens(" ".join(current_chunk))

            if current_size >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-50:]
                current_size = count_tokens(" ".join(current_chunk))

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        self.chunks = chunks
        return chunks

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def create_index(self, texts: List[str]):
        embeddings = cohere_client.embed(
            texts=texts,
            model='embed-english-v3.0',
            input_type='search_query'
        ).embeddings

        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts = texts

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        query_embedding = cohere_client.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type='search_query'
        ).embeddings[0]

        D, I = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )

        results = [(self.texts[i], D[0][idx]) for idx, i in enumerate(I[0])]
        return results

def generate_response(query: str, context: str, max_tokens: int = 4000) -> str:
    base_prompt = """You are a helpful assistant. Use the following context to answer the question. 
    If you cannot answer based on the context, say you don't know.

    Context: {context}

    Question: {query}

    Answer:"""

    prompt_tokens = count_tokens(base_prompt.format(context="", query=query))
    available_tokens = max_tokens - prompt_tokens - 1024

    context_tokens = count_tokens(context)
    if context_tokens > available_tokens:
        encoding = tiktoken.get_encoding("cl100k_base")
        context_ids = encoding.encode(context)
        truncated_ids = context_ids[:available_tokens]
        context = encoding.decode(truncated_ids)

    prompt = base_prompt.format(context=context, query=query)

    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            temperature=0.3,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --- Streamlit UI with Custom HTML & CSS ---

st.set_page_config(page_title="📚 PDF Chat Assistant", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .sidebar .sidebar-content { padding: 20px; }
    .main-content { background-color: #f9f9f9; padding: 20px; border-radius: 10px; }
    .chat-bubble-user {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: right;
    }
    .chat-bubble-assistant {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: left;
        border: 1px solid #ddd;
    }
    .chat-input {
        padding: 10px;
        width: 100%;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .send-btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📄 Upload & Instructions")
st.sidebar.info("""
1. Upload your PDF file.
2. Wait for processing to complete.
3. Ask any question related to the document in the chat.
""")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

st.title("🤖 PDF Chat Assistant")
st.write("Chat with your document as if you're chatting with a human assistant!")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'messages' not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            processor = DocumentProcessor(chunk_size=500)
            chunks = processor.process_pdf(uploaded_file)
            st.session_state.vector_store.create_index(chunks)
            st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"Document processing failed: {str(e)}")

# Chat History Display
st.subheader("Chat")
chat_placeholder = st.container()
with chat_placeholder:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-assistant">{message["content"]}</div>', unsafe_allow_html=True)

# Chat Input
query = st.text_input("Type your message:", key="input")
send_button = st.button("Send")

if send_button and query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        if st.session_state.vector_store.index is not None:
            try:
                results = st.session_state.vector_store.search(query, k=2)
                context = "\n".join([chunk for chunk, _ in results])
                response = generate_response(query, context)
            except Exception as e:
                response = f"Error: {str(e)}"
        else:
            response = "❗ Please upload a document first!"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
