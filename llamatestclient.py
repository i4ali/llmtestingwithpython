import pytest
import requests
import time


class OllamaTestClient:
    def __init__(self, model="tinyllama", host="localhost", port=11434):
        self.model = model
        self.endpoint = f"http://{host}:{port}/api/generate"

    def check_model_available(self):
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": "test",
                "stream": False
            })
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt, temperature=0.8, max_tokens=1000):
        start_time = time.time()
        response = requests.post(self.endpoint, json={
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        })
        end_time = time.time()
        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")
        result = response.json()
        return {
            "text": result["response"],
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(result["response"].split()),
            "latency": end_time - start_time
        }


@pytest.fixture
def llm_client():
    client = OllamaTestClient(model="tinyllama")
    if not client.check_model_available():
        pytest.skip("Tinyllama model is not available in Ollama. Please run 'Ollama pull tinyllama' first")

    return client
