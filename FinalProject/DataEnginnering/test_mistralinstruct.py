import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()  # charge le .env et donc HUGGINGFACE_HUB_TOKEN

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
print("Token trouvé :", token is not None)  # juste pour vérifier

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

pipe = pipeline(
    "text2text-generation",
    model=model_name,
    device_map="auto",
    use_auth_token=token  # Passe ton token ici explicitement
)

res = pipe("Bonjour, comment vas-tu ?")
print(res)
