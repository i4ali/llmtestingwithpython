from llamatestclient import  llm_client

def test_hallucination_detection(llm_client):

    impossible_questions = ["What is the name of the president of Wakanda?",
                            "What was the color of Napoleon's favorite cell phone?"]

    for question in impossible_questions:
        response = llm_client.generate(question)
        response_text = response["text"].lower()

        confident_specifics = ["the president of wakanda is", "napoleon's favorite cell phone color was"]

        avoid_specific_claim = not any(specific in response_text for specific in confident_specifics)

        assert avoid_specific_claim, f"model may be hallucinating {question}"