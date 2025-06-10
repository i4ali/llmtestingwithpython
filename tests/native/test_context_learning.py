from llamatestclient import llm_client


def test_logical_consistency(llm_client):

    equivalent_questions = [("What is the capital of Japan?", "Tokyo is the capital of which country?")]

    for q1, q2 in equivalent_questions:
        response1 = llm_client.generate(q1)
        response2 = llm_client.generate(q2)

        key_info1 = response1["text"].lower()[:100]
        key_info2 = response2["text"].lower()[:100]

        words1 = set(key_info1.split())
        print(words1)
        words2 = set(key_info2.split())
        print(words2)

        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            overlap_ratio = overlap / union

            assert overlap_ratio > 0.2, f"Inconsistent answers for equivalent questions: '{q1}' and '{q2}"

def test_error_handling(llm_client):
    error_inputs = ["What is the color of silence?", "how many corners does a circle have?"]

    for prompt in error_inputs:
        response = llm_client.generate(prompt)

        assert len(response["text"]) > 20, "Response too short for error input"

        error_handling_indicators = ["undefined", "doesn't have", "doesn't make sense", "nonsensical", "understand", "concept"]

        has_indicator = any(indicator in response["text"].lower() for indicator in error_handling_indicators)

        assert has_indicator, f"Model didnt properly handle erroneous inputs: '{prompt}'"









