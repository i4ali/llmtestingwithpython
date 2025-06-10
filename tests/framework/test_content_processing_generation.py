from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4o-mini")
evaluation_model = GPTModel(model="gpt-4o")

def test_list_format():
    """Test if the LLM can generate content in a numbered list format"""
    prompt = "List 3 fruits. Format your answer as a numbered list."
    response_text = openai_client.generate(prompt)

    # DeepEval metric for list formatting
    list_format_metric = GEval(
        name="List Format Compliance",
        criteria="""Evaluate if the response properly follows the numbered list format request:
        1. Contains exactly 3 items (fruits)
        2. Uses numbered list format (1., 2., 3. or 1), 2), 3))
        3. Each item should be a recognizable fruit name
        4. Proper formatting with clear separation between items

        Award full points only if all formatting requirements are met.""",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=evaluation_model
    )

    test_case = LLMTestCase(
        input=prompt,
        actual_output=response_text["text"]
    )

    assert_test(test_case, [list_format_metric])
    print(f"Response: {response_text}")


def test_simple_json():
    """Test if the LLM can generate valid JSON with specific content"""
    prompt = "Create a JSON with a person's name and age. Use the name John and age 30."
    response_text = openai_client.generate(prompt)

    # DeepEval metric for JSON generation
    json_format_metric = GEval(
        name="JSON Format and Content",
        criteria="""Evaluate if the response meets JSON generation requirements:
        1. Contains valid JSON structure with proper braces { }
        2. Includes a 'name' field with the value 'John'
        3. Includes an 'age' field with the value 30
        4. Uses proper JSON syntax (quotes, colons, commas)
        5. The JSON should be parseable and well-formatted

        Deduct points for invalid JSON syntax or missing required fields.""",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=evaluation_model
    )

    test_case = LLMTestCase(
        input=prompt,
        actual_output=response_text["text"]
    )

    assert_test(test_case, [json_format_metric])
    print(f"Response: {response_text}")