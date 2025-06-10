import time
from llamatestclient import llm_client

def test_basic_throughput(llm_client):
    prompt = "hello"

    start_time = time.time()
    responses = []

    for _ in range(3):
        response = llm_client.generate(prompt, max_tokens=20)
        responses.append(response)

    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = sum(r["completion_tokens"] for r in responses)
    tokens_per_second = total_tokens / total_time
    requests_per_minute = (3 / total_time) * 60

    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"Request rate: {requests_per_minute:.2f} requests/minute")
    print(f"Average response size: {total_tokens/3:.2f} tokens")




