import subprocess
import json

def extract_personality_traits(personalities):
    prompt = f"""
You are given a list of personality descriptions.
Extract ONLY personality traits (character-related qualities).
Exclude interests, hobbies, personal tastes, professions, physical descriptions, or preferences.

Return the result as a JSON list of lowercase strings.

List:
{json.dumps(personalities, ensure_ascii=False)}
"""

    cmd = ["ollama", "run", "mistral"]

    # On envoie le prompt via stdin
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr}")

    output = result.stdout.strip()

    try:
        traits = json.loads(output)
        return traits
    except json.JSONDecodeError:
        # Si la sortie n'est pas un JSON pur, on essaie d'extraire la partie JSON
        # ou on affiche un message d'erreur plus clair
        raise ValueError(f"Impossible de parser la sortie JSON : {output}")

# Exemple
personalities = [
  "passionate about music",
  "optimistic",
  "loves to sing and play the guitar",
  "always has a smile on her face"
]

traits = extract_personality_traits(personalities)
print("Traits extraits:", traits)
