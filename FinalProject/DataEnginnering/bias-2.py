import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def load_character_genders(characters_path):
    gender_map = {}
    with open(characters_path, encoding="latin1") as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) < 6:
                # Optionnel : afficher ligne incorrecte
                print(f"Ligne inattendue (moins de 6 champs) : {line.strip()}")
                continue
            character_id = parts[0]
            gender = parts[4]  # ici 5e champ (index 4) est le genre
            gender_map[character_id] = gender
    return gender_map


def normalize_gender_key(g):
    if g is None:
        return "?"
    g_lower = g.lower()
    if g_lower == 'm':
        return 'm'
    if g_lower == 'f':
        return 'f'
    return g_lower if g_lower else "?"


def clean_gender_counts(d):
    new_counter = defaultdict(int)
    for k, v in d.items():
        new_key = normalize_gender_key(k)
        new_counter[new_key] += v
    return dict(new_counter)


def analyze_romantic_conversations(convo_file, name_to_gender):
    """Analyse le corpus romantique pour détecter les biais"""
    total_conversations = 0
    total_turns = 0
    total_words = 0
    gender_pair_counts = Counter()
    speaker_turn_counts = Counter()
    speaker_word_counts = defaultdict(int)
    first_speaker_counts = Counter()
    last_speaker_counts = Counter()
    conversation_lengths = []

    with open(convo_file, 'r', encoding='utf-8') as f:
        for line in f:
            convo = json.loads(line)
            if not convo or len(convo["lines"]) < 2:
                continue

            # Extraire les character_ids des speakers
            character_ids = list({turn["character_id"] for turn in convo["lines"]})
            if len(character_ids) != 2:
                continue

            total_conversations += 1
            conversation_lengths.append(len(convo["lines"]))
            total_turns += len(convo["lines"])

            g1 = normalize_gender_key(name_to_gender.get(character_ids[0], "?"))
            g2 = normalize_gender_key(name_to_gender.get(character_ids[1], "?"))
            gender_pair = tuple(sorted([g1, g2]))
            gender_pair_counts[gender_pair] += 1

            # Premier et dernier locuteur (character_id)
            first = convo["lines"][0]["character_id"]
            last = convo["lines"][-1]["character_id"]
            first_speaker_counts[normalize_gender_key(name_to_gender.get(first, "?"))] += 1
            last_speaker_counts[normalize_gender_key(name_to_gender.get(last, "?"))] += 1

            # Comptage des mots par genre, par character_id
            for turn in convo["lines"]:
                char_id = turn["character_id"]
                text = turn["text"]
                word_count = len(re.findall(r'\w+', text))
                speaker_turn_counts[char_id] += 1
                speaker_word_counts[char_id] += word_count
                total_words += word_count

    # Moyenne de mots par réplique, par genre
    gender_word_totals = Counter()
    gender_turn_totals = Counter()
    for speaker, turns in speaker_turn_counts.items():
        gender = normalize_gender_key(name_to_gender.get(speaker, "?"))
        gender_turn_totals[gender] += turns
        gender_word_totals[gender] += speaker_word_counts[speaker]

    avg_words_per_turn = {
        gender: gender_word_totals[gender] / gender_turn_totals[gender]
        for gender in gender_turn_totals
    }

    # Résumé
    summary = {
        "total_conversations": total_conversations,
        "avg_turns_per_conversation": total_turns / total_conversations,
        "avg_words_per_turn_by_gender": avg_words_per_turn,
        "gender_pair_counts": dict(gender_pair_counts),
        "first_speaker_gender": dict(first_speaker_counts),
        "last_speaker_gender": dict(last_speaker_counts),
        "conversation_lengths": conversation_lengths,
    }

    return summary


def plot_stats(summary):
    """Affiche des graphes pour le rapport"""
    # Genre pair counts
    plt.figure(figsize=(6, 4))
    labels = [f"{a}/{b}" for (a, b) in summary["gender_pair_counts"]]
    values = list(summary["gender_pair_counts"].values())
    plt.bar(labels, values)
    plt.title("Nombre de conversations par combinaison de genres")
    plt.ylabel("Nombre de conversations")
    plt.xlabel("Genre 1 / Genre 2")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Average words per turn
    plt.figure(figsize=(6, 4))
    labels = list(summary["avg_words_per_turn_by_gender"].keys())
    values = list(summary["avg_words_per_turn_by_gender"].values())
    plt.bar(labels, values)
    plt.title("Mots moyens par réplique selon le genre")
    plt.ylabel("Mots / réplique")
    plt.xlabel("Genre")
    plt.tight_layout()
    plt.show()

    # First/last speaker
    for label, data in [("Premiers à parler", summary["first_speaker_gender"]),
                        ("Derniers à parler", summary["last_speaker_gender"])]:
        plt.figure(figsize=(6, 4))
        genders = list(data.keys())
        counts = list(data.values())
        plt.bar(genders, counts)
        plt.title(label)
        plt.ylabel("Nombre de conversations")
        plt.xlabel("Genre")
        plt.tight_layout()
        plt.show()


def main():
    characters_path = "archive/movie_characters_metadata.txt"
    conversations_path = "romance_conversations.jsonl"

    print("🔍 Chargement des genres des personnages...")
    gender_map = load_character_genders(characters_path)

    print("📊 Analyse des conversations romantiques...")
    summary = analyze_romantic_conversations(conversations_path, gender_map)

    print("\n📄 Résumé des statistiques :")
    print(f"  - Total de conversations : {summary['total_conversations']}")
    print(f"  - Tours moyens / conversation : {summary['avg_turns_per_conversation']:.2f}")
    print("  - Mots moyens / réplique (par genre) :")
    for g, avg in summary["avg_words_per_turn_by_gender"].items():
        print(f"      {g}: {avg:.2f} mots")
    print("  - Combinaisons de genres :")
    for pair, count in summary["gender_pair_counts"].items():
        print(f"      {pair[0]} / {pair[1]} : {count}")
    print("  - Genre du premier locuteur :")
    for g, count in summary["first_speaker_gender"].items():
        print(f"      {g}: {count}")
    print("  - Genre du dernier locuteur :")
    for g, count in summary["last_speaker_gender"].items():
        print(f"      {g}: {count}")

    print("\n📈 Génération des graphiques...")
    plot_stats(summary)


if __name__ == "__main__":
    main()
