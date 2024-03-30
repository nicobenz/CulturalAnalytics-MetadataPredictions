import json
from tqdm import tqdm


def find_top_level_genre_id(genre_id, genres):
    """Recursively find the top-level genre id for a given subgenre id."""
    parent_genre = genres.get(genre_id, {}).get("subgenre of")
    if parent_genre:
        # Recursively trace back to the top-level genre
        return find_top_level_genre_id(parent_genre[0]["id"], genres)
    else:
        # Top-level genre found
        return genre_id


with open("/Volumes/Data/bothbrainz/finalized_data.json") as f:
    content = json.load(f)

with open("data/genres.json") as f:
    genres = json.load(f)

genre_added = []
for k, v in content.items():
    subgenre_id = v["genre"]
    top_level_genre_id = find_top_level_genre_id(subgenre_id, genres)
    if top_level_genre_id in genres:
        v["top level genre id"] = top_level_genre_id
        v["top level genre name"] = genres[top_level_genre_id]["name"]
        v["album id"] = k
        genre_added.append(v)


subgenre_moods = {}
genre_moods = {}

# Assuming genre_added is your list of dictionaries representing music albums

for item in tqdm(genre_added):
    if item["top level genre id"] not in genre_moods:
        genre_moods[item["top level genre id"]] = {
            "acoustic": item["acoustic"],
            "aggressive": item["aggressive"],
            "electronic": item["electronic"],
            "happy": item["happy"],
            "party": item["party"],
            "relaxed": item["relaxed"],
            "sad": item["sad"],
            "name": item["top level genre name"],
            "count": 1  # Initialize count to 1 for first occurrence
        }
    else:
        genre_moods[item["top level genre id"]]["acoustic"] += item["acoustic"]
        genre_moods[item["top level genre id"]]["aggressive"] += item["aggressive"]
        genre_moods[item["top level genre id"]]["electronic"] += item["electronic"]
        genre_moods[item["top level genre id"]]["happy"] += item["happy"]
        genre_moods[item["top level genre id"]]["party"] += item["party"]
        genre_moods[item["top level genre id"]]["relaxed"] += item["relaxed"]
        genre_moods[item["top level genre id"]]["sad"] += item["sad"]
        genre_moods[item["top level genre id"]]["count"] += 1  # Increment count

    if item["genre"] not in subgenre_moods:
        subgenre_moods[item["genre"]] = {
            "acoustic": item["acoustic"],
            "aggressive": item["aggressive"],
            "electronic": item["electronic"],
            "happy": item["happy"],
            "party": item["party"],
            "relaxed": item["relaxed"],
            "sad": item["sad"],
            "name": item["genre_name"],
            "count": 1  # Initialize count to 1 for first occurrence
        }
    else:
        subgenre_moods[item["genre"]]["acoustic"] += item["acoustic"]
        subgenre_moods[item["genre"]]["aggressive"] += item["aggressive"]
        subgenre_moods[item["genre"]]["electronic"] += item["electronic"]
        subgenre_moods[item["genre"]]["happy"] += item["happy"]
        subgenre_moods[item["genre"]]["party"] += item["party"]
        subgenre_moods[item["genre"]]["relaxed"] += item["relaxed"]
        subgenre_moods[item["genre"]]["sad"] += item["sad"]
        subgenre_moods[item["genre"]]["count"] += 1  # Increment count

mean_genre_moods = {}
mean_subgenre_moods = {}

for genre_id, mood_dict in genre_moods.items():
    count = mood_dict.pop("count")  # Remove count from mood_dict
    name = mood_dict.pop("name")  # Remove name from mood_dict
    mean_genre_moods[name] = {k: v / count for k, v in mood_dict.items()}

for subgenre_id, mood_dict in subgenre_moods.items():
    count = mood_dict.pop("count")  # Remove count from mood_dict
    name = mood_dict.pop("name")  # Remove name from mood_dict
    mean_subgenre_moods[name] = {k: v / count for k, v in mood_dict.items()}

with open("results/genre_moods.json", "w") as f:
    json.dump(mean_genre_moods, f, indent=4, sort_keys=True)

with open("results/subgenre_moods.json", "w") as f:
    json.dump(mean_subgenre_moods, f, indent=4, sort_keys=True)
