from llamatestclient import llm_client
import re

def test_list_format(llm_client):
    prompt = "List 3 fruits. Format your answer as a numbered list."

    response = llm_client.generate(prompt)

    list_pattern = r'[1-3][.)]'
    matches = re.findall(list_pattern, response["text"])

    assert len(matches) >= 1,  "Response doesnt contain a numbered list format"
    print(f"Found {len(matches)} list markers in response")
    print(f"Response: {response['text']}")

def test_simple_json(llm_client):
    prompt = "Create a JSON with a person's name and age. Use the name John and age 30."

    response = llm_client.generate(prompt)

    contains_braces = "{" in response["text"] and "}" in response["text"]
    contains_name = "name" in response["text"].lower() and "john" in response["text"].lower()
    contains_age = "age" in response["text"].lower() and "30" in response["text"]

    assert contains_braces, "Response doesnt contain JSON braces"
    assert contains_name, "Response doesnt mention name"
    assert contains_age, "Response doesnt mention age"

    print(f"Response: {response['text']}")

