from langchain_google_genai import GoogleGenerativeAI

class Gemini:
    def __init__(self, model_name):
        self.model = GoogleGenerativeAI(model_name=model_name, temperature=0.7)
