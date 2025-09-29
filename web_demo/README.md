# Shoe Store RAG Chatbot Web Demo

A simple Flask web application demonstrating RAG (Retrieval-Augmented Generation) functionality using the shoe store knowledge base.

## Features

- **Real-time chat interface** with the RAG-powered chatbot
- **Context visualization** showing what documents were retrieved from Pinecone
- **Demo questions** to quickly test different scenarios
- **Responsive design** that works on desktop and mobile

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export PINECONE_API_KEY="your_pinecone_key"
   export OPENAI_API_KEY="your_openai_key"
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser to:**
   ```
   http://localhost:5000
   ```

## How it Works

1. **User asks a question** through the web interface
2. **RAG system retrieves** relevant context from Pinecone vector database
3. **OpenAI GPT-4o generates** a response using the retrieved context
4. **Interface shows both** the bot response and the retrieved documents

## Demo Questions

Try these to see the RAG system in action:
- "What is your return policy?"
- "Do you offer student discounts?"
- "What shoe brands do you carry?"
- "How long does shipping take?"

The retrieved context panel will show you exactly which documents from the knowledge base were used to generate each response!