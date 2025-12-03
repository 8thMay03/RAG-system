from langchain_google_genai import GoogleGenerativeAI


class GeminiFlash:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = GoogleGenerativeAI(model=model_name, temperature=0.7)

    def get_model(self):
        return self.model

class GeminiPro:
    def __init__(self, model_name="gemini-2.5-pro"):
        self.model = GoogleGenerativeAI(model=model_name, temperature=0.7)

    def get_model(self):
        return self.model