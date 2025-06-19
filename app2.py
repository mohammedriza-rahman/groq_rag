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
#                 current_chunk = current_chunk[-50:]
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

# # --- Streamlit UI with Custom HTML & CSS ---

# st.set_page_config(page_title="📚 PDF Chat Assistant", page_icon="🤖", layout="wide")

# # Custom CSS
# st.markdown("""
#     <style>
#     .sidebar .sidebar-content { padding: 20px; }
#     .main-content { background-color: #f9f9f9; padding: 20px; border-radius: 10px; }
#     .chat-bubble-user {
#         background-color: #DCF8C6;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 10px 0;
#         text-align: right;
#     }
#     .chat-bubble-assistant {
#         background-color: #FFFFFF;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 10px 0;
#         text-align: left;
#         border: 1px solid #ddd;
#     }
#     .chat-input {
#         padding: 10px;
#         width: 100%;
#         border-radius: 5px;
#         border: 1px solid #ccc;
#     }
#     .send-btn {
#         background-color: #4CAF50;
#         color: white;
#         padding: 10px 20px;
#         border: none;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.sidebar.title("📄 Upload & Instructions")
# st.sidebar.info("""
# 1. Upload your PDF file.
# 2. Wait for processing to complete.
# 3. Ask any question related to the document in the chat.
# """)
# uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

# st.title("🤖 PDF Chat Assistant")
# st.write("Chat with your document as if you're chatting with a human assistant!")

# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = VectorStore()
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# if uploaded_file:
#     with st.spinner("Processing document..."):
#         try:
#             processor = DocumentProcessor(chunk_size=500)
#             chunks = processor.process_pdf(uploaded_file)
#             st.session_state.vector_store.create_index(chunks)
#             st.success("✅ Document processed successfully!")
#         except Exception as e:
#             st.error(f"Document processing failed: {str(e)}")

# # Chat History Display
# st.subheader("Chat")
# chat_placeholder = st.container()
# with chat_placeholder:
#     for message in st.session_state.messages:
#         if message["role"] == "user":
#             st.markdown(f'<div class="chat-bubble-user">{message["content"]}</div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="chat-bubble-assistant">{message["content"]}</div>', unsafe_allow_html=True)

# # Chat Input
# query = st.text_input("Type your message:", key="input")
# send_button = st.button("Send")

# if send_button and query:
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.spinner("Thinking..."):
#         if st.session_state.vector_store.index is not None:
#             try:
#                 results = st.session_state.vector_store.search(query, k=2)
#                 context = "\n".join([chunk for chunk, _ in results])
#                 response = generate_response(query, context)
#             except Exception as e:
#                 response = f"Error: {str(e)}"
#         else:
#             response = "❗ Please upload a document first!"

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.rerun()

import streamlit as st
import os
from typing import List, Tuple
import PyPDF2
from io import BytesIO
import faiss
import numpy as np
from groq import Groq
from cohere import Client
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
                current_chunk = current_chunk[-50:]  # overlap
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

# --- Streamlit UI ---

# Custom CSS for chat interface
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
    background-color: #f5f5f5;
}
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.header {
    display: flex;
    align-items: center;
    gap: 15px;
    padding-bottom: 20px;
    border-bottom: 1px solid #e5e5e5;
}
.header img {
    width: 40px;
    height: 40px;
}
.header h1 {
    font-size: 24px;
    color: #1a1a1a;
}
.sidebar .sidebar-content {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 8px;
}
.stFileUploader {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e5e5e5;
}
.chat-container {
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    background-color: #fafafa;
    margin-bottom: 20px;
}
.chat-message {
    margin-bottom: 15px;
    padding: 10px 15px;
    border-radius: 8px;
    max-width: 70%;
}
.user-message {
    background-color: #007bff;
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 2px;
}
.assistant-message {
    background-color: #e5e5e5;
    color: #1a1a1a;
    margin-right: auto;
    border-bottom-left-radius: 2px;
}
.stTextInput > div > div > input {
    border-radius: 20px;
    padding: 10px 15px;
}
.footer {
    text-align: center;
    color: #666;
    font-size: 14px;
    padding-top: 20px;
    border-top: 1px solid #e5e5e5;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload and instructions
with st.sidebar:
    st.markdown("<h2 style='color: #1a1a1a;'>📚 Document Chatbot</h2>", unsafe_allow_html=True)
    st.markdown("""
    **How to use:**
    1. Upload a PDF document using the uploader below.
    2. Wait for the document to process.
    3. Ask questions about the document in the chat input.
    4. Get contextual answers based on your PDF!
    
    *Note:* Ensure your PDF contains text (not scanned images) for best results.
    """)
    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf", help="Upload a PDF to start chatting with its content.")

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <img src="https://via.placeholder.com/40" alt="Logo">
    <h1>Document Chat Assistant</h1>
</div>
""", unsafe_allow_html=True)

# Process uploaded file
if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            processor = DocumentProcessor(chunk_size=500)
            chunks = processor.process_pdf(uploaded_file)
            st.session_state.vector_store.create_index(chunks)
            st.success("✅ Document processed! Start asking questions.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if query := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="chat-message user-message">{query}</div>', unsafe_allow_html=True)

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
        st.markdown(f'<div class="chat-message assistant-message">{response}</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Powered by xAI & Streamlit</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
