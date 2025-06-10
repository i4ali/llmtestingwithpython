import os

from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.framework.shoe_store_rag import ShoeStoreRAG


def test_shoe_store_with_deepeval():

    pinecone_key = os.getenv('PINECONE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')

    rag = ShoeStoreRAG(pinecone_api_key=pinecone_key, openai_api_key=openai_key)

    test_scenarios = [
        "What if these shoes don't fit?",
        "Do you offer student discounts?",
        "How long does shipping take?",
        "What brands do you carry"
    ]

    test_cases = []
    for query in test_scenarios:
        retrieval_context = rag.retrieve_context(query=query)
        actual_output = rag.generate_answer(query, retrieval_context)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)



    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    evaluate(test_cases=test_cases, metrics=[contextual_relevancy_metric])

