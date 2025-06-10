from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient(model="gpt-4o-mini")
evaluation_model = GPTModel(model="gpt-4o")

def test_basic_response():
    prompt = "Hello, how are you?"
    response_text = openai_client.generate(prompt)

    test_case = LLMTestCase(input=prompt, actual_output=response_text["text"])

    response_quality_metric = GEval(name="Response Quality",
          criteria="Determine if the response is substantial and appropriate. The response should be non-empty, have more than 10 characters, and be a reasonable greeting response",
          evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
          threshold=0.7,
          model=evaluation_model
          )

    assert_test(test_case, [response_quality_metric])


def test_instruction_following():
    """Test that the model can follow simple instructions to list colors."""
    prompt = "Name three colors. Just list the colors"
    response = openai_client.generate("Name three colors. Just list the colors")

    test_case = LLMTestCase(
        input=prompt,
        actual_output=response["text"]
    )

    instruction_following_metric = GEval(
        name="instruction following",
        criteria="Determine if the response correctly follows the instruction to list colors. The response should contain atleast one recognizable color name",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=evaluation_model
    )

    assert_test(test_case, [instruction_following_metric])


def test_simple_qa():
    """Test basic Q&A functionality with a factual question."""
    prompt = "what is the capital of France?"
    response = openai_client.generate("What is the capital of France?")

    factual_metric = GEval(
        name="simple qa",
        criteria="Determine if the response correctly identifies Paris as the capital of France",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.8,
        model=evaluation_model
    )

    test_case = LLMTestCase(
        input=prompt,
        actual_output=response["text"],
        expected_output="The capital of France is Paris"
    )

    assert_test(test_case, [factual_metric])



def test_multi_turn_basic():
    """Test if the model can remember information from previous context."""
    prompt = """
    User: My name is Sam.
    Assistant: Hello Sam! How can I help you?
    User: What's my name?
    """

    response = openai_client.generate(prompt)




