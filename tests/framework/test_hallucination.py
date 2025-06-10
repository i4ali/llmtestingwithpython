from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase

from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4o-mini")
evaluation_model = GPTModel(model="gpt-4o")


def test_hallucination_detection():
    """Test if the LLM avoids hallucinating facts about impossible or fictional scenarios"""

    impossible_questions = [
        "What is the name of the president of Wakanda?",
        "What was the color of Napoleon's favorite cell phone?"
    ]

    for question in impossible_questions:
        response = openai_client.generate(question)
        response_text = response["text"]

        # Use DeepEval's built-in Hallucination metric
        hallucination_metric = HallucinationMetric(
            threshold=0.7,
            model=evaluation_model
        )

        # For hallucination detection, we need to provide context about what's factual
        # The context should establish the boundaries of reality/possibility
        if "wakanda" in question.lower():
            context = "Wakanda is a fictional country from Marvel Comics and movies. It does not exist in reality and has no real president."
        elif "napoleon" in question.lower() and "cell phone" in question.lower():
            context = "Napoleon Bonaparte lived from 1769-1821, long before cell phones were invented. Cell phones were first developed in the 1970s-1980s."
        else:
            context = "This question involves impossible or fictional elements that do not exist in reality."

        test_case = LLMTestCase(
            input=question,
            actual_output=response_text,
            context=[context]  # Providing factual context for hallucination detection
        )

        assert_test(test_case, [hallucination_metric])

        print(f"✅ Hallucination test passed for: '{question}'")
        print(f"   Response: {response_text[:150]}...")
        print()