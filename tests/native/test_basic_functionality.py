from llamatestclient import llm_client


def test_basic_response(llm_client):
    response = llm_client.generate("Hello, how are you?")

    assert response['text'].strip() != ""
    assert len(response['text']) > 10, "response is too short"

    print(f"Basic response test - response length: {len(response['text'])}")


def test_instruction_following(llm_client):
    response = llm_client.generate("Name three colors. Just list the colors")
    response_text = response["text"].lower()

    colors = ["red", "blue", "green", "yellow", "orange", "purple", "black"
                                                                    "white", "pink", "brown", "gray"]

    color_count = sum(1 for color in colors if color in response_text)

    assert color_count >= 1, f"Expected at least 1 color, found {color_count}"
    print(f"Colors found in response: {color_count}")

def test_simple_qa(llm_client):
    response = llm_client.generate("what is the capital of France?")
    assert "paris" in response["text"].lower() or "Paris" in response["text"], "Failed to correctly identify " \
                                                                               "Paris as the capital of France"

def test_multi_turn_basic(llm_client):

    prompt = """
    User: My name is Sam.
    Assistant: Hello Sam! How can I help you?
    User: Whats my name?
    """

    response = llm_client.generate(prompt)
    assert "sam" in response["text"].lower() or "Sam" in response["text"], "Model failed to remember the name from " \
                                                                           "previous message"




