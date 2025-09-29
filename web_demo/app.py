import os
from flask import Flask, render_template, request, jsonify

from shoe_store_rag import ShoeStoreRAG

app = Flask(__name__)

rag = None

def initialize_rag():
    global rag
    if rag is None:
        pinecone_key = os.getenv('PINECONE_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if not pinecone_key or not openai_key:
            raise ValueError("Missing required environment variables: PINECONE_API_KEY and OPENAI_API_KEY")
        
        rag = ShoeStoreRAG(pinecone_api_key=pinecone_key, openai_api_key=openai_key)

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        initialize_rag()
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        context = rag.retrieve_context(user_message)
        response = rag.generate_answer(user_message, context)
        
        return jsonify({
            'response': response,
            'context': context
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)