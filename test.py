import google.generativeai as genai
genai.configure(api_key="AIzaSyBwDL4PXHqzmHrRQBRCqD74oTC3XU0P4-g")
models = genai.list_models()

for model in models:
    print(model.name)