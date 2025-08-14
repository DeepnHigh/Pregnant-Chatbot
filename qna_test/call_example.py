import requests

url = "http://192.168.0.21:8081/v1/chat/completions"

payload = {
    "messages": [
        {"role": "user", "content": "안녕하세요. 자기소개 좀 해줘"}
    ]
}

response = requests.post(url, json=payload)
print(response.json())
