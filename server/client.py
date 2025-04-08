import requests

# server url
server_url = "http://localhost:8000/generate"

# request prompts
data = {
    "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
}

# POST to server
response = requests.post(server_url, json=data)

# Getting Response
if response.status_code == 200:
    responses = response.json().get("responses", [])
    for idx, text in enumerate(responses):
        print(f"Response {idx + 1}: {text}")
else:
    print(f"Error: {response.status_code}")
