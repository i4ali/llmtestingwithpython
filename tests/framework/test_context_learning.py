from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4o-mini")
evaluation_model = GPTModel(model="gpt-4o")


def test_logical_consistency():
    """Test if the LLM provides consistent answers to equivalent questions"""

    # Test equivalent question pairs
    equivalent_questions = [
        ("What is the capital of Japan?", "Tokyo is the capital of which country?"),
    ]

    for q1, q2 in equivalent_questions:
        response1 = openai_client.generate(q1)
        response2 = openai_client.generate(q2)

        # DeepEval metric for logical consistency
        consistency_metric = GEval(
            name="Logical Consistency",
            criteria="""Evaluate whether these two responses demonstrate logical consistency:
            1. Core factual information should be the same in both responses
            2. Both responses should address the same underlying concept
            3. No contradictory information between the responses
            4. The essential answer/conclusion should align across both responses
            5. Allow for different phrasing/style but require factual consistency

            Award high scores when both responses contain the same key facts, even if expressed differently.
            Deduct points for any contradictory or inconsistent information.""",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.8,  # High threshold for consistency requirement
            model=evaluation_model
        )

        # Create a combined test case that includes both Q&A pairs
        response1 = response1["text"]
        response2 = response2["text"]
        combined_input = f"Question 1: {q1}\nQuestion 2: {q2}"
        combined_output = f"Response 1: {response1}\nResponse 2: {response2}"

        test_case = LLMTestCase(
            input=combined_input,
            actual_output=combined_output
        )

        assert_test(test_case, [consistency_metric])

        print(f"✅ Consistency test passed for: '{q1}' & '{q2}'")
        print(f"   Response 1: {response1[:100]}...")
        print(f"   Response 2: {response2[:100]}...")
        print()


def test_error_handling():
    """Test if the LLM properly handles ambiguous, nonsensical, or impossible questions"""

    error_inputs = [
        "What is the color of silence?",
        "How many corners does a circle have?"
    ]

    for prompt in error_inputs:
        response = openai_client.generate(prompt)
        response_text = response["text"]

        # DeepEval metric for error handling
        error_handling_metric = GEval(
            name="Error Handling Quality",
            criteria="""Evaluate how well the response handles this ambiguous/nonsensical question:
            1. Recognition: Acknowledges that the question is ambiguous, nonsensical, or unanswerable
            2. Helpfulness: Offers alternative perspectives or related information when appropriate
            3. Tone: Maintains respectful, helpful tone without being dismissive
            4. Completeness: Provides sufficient explanation (not just "I don't know")

            """,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.7,
            model=evaluation_model
        )

        test_case = LLMTestCase(
            input=prompt,
            actual_output=response_text
        )

        assert_test(test_case, [error_handling_metric])

        print(f"✅ Error handling test passed for: '{prompt}'")
        print(f"   Response: {response_text[:150]}...")
        print()