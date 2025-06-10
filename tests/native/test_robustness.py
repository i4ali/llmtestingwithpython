from llamatestclient import llm_client

def test_input_variations(llm_client):
    variations = ["what is the capital of france?", "What is the capital of France?", "capital of france?"]

    for prompt in variations:
        response = llm_client.generate(prompt)
        contains_paris = "paris" in response["text"].lower() or "Paris" in response["text"]

        assert contains_paris, f"output does not contain word paris {prompt}"