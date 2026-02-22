import google.generativeai as genai

genai.configure(api_key="AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA")

print("--- Available Models for You ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Name: {m.name}")