from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric


evaluation_model = GPTModel(model="gpt-4o")

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

# Replace this with the actual retrieved context from your RAG pipeline
# retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]
retrieval_context = ["Our stores close after 5pm everyday"]

context_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=evaluation_model,
    include_reason=True
)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    retrieval_context=retrieval_context
)

evaluate(test_cases=[test_case], metrics=[context_relevancy_metric])