import os
import shutil
import json
import ijson
import time
import uuid
from tqdm import tqdm
from icecream import ic
from collections import Counter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import squarify
import random



def get_all_files():
    file_dir = "/Volumes/Data/acousticbrainz_untared/acousticbrainz-highlevel-json-20220623/highlevel"
    file_list = []
    for root, _, dir_files in tqdm(os.walk(file_dir), total=4353):
        for subfile in dir_files:
            if not subfile.endswith(".json"):
                continue
            file_path = os.path.join(root, subfile)
            file_list.append(file_path)

    return file_list


#path = "/Volumes/Data/acousticbrainz_untared/acousticbrainz-highlevel-json-20220623/highlevel/00/0"
#test_file = f"{path}/0000a1d3-7b4a-4a74-92c6-adb44393e41e-0.json"
root_path = "/Volumes/Data/acousticbrainz_untared/acousticbrainz-highlevel-json-20220623/highlevel/"

"""
files = get_all_files()
files = [file.replace(root_path, "") for file in files]
with open("data/all_files.txt", "w") as f:
    for file in files:
        f.write(f"{file}\n")
time.sleep(10)
"""


def is_valid_uuid(uuid_to_test):
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=3)
        if not str(uuid_obj) == uuid_to_test:
            uuid_obj = uuid.UUID(uuid_to_test, version=4)
            return str(uuid_obj) == uuid_to_test
        return str(uuid_obj) == uuid_to_test
    except ValueError:
        return False


def extract_moods():
    all_files = "data/all_files.txt"
    file_size = os.path.getsize(all_files)

    files = []
    with open(all_files) as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading all file paths to list") as pbar:
            for line in f:
                files.append(f"{root_path}{line.strip()}")
                pbar.update(len(line.encode('utf-8')))

    mood_labels = [
        "mood_acoustic",
        "mood_aggressive",
        "mood_electronic",
        "mood_happy",
        "mood_party",
        "mood_relaxed",
        "mood_sad"
    ]

    musicbrainz_labels = [
        "musicbrainz_albumartistid",
        "musicbrainz_albumid",
        "musicbrainz_artistid",
        "musicbrainz_recordingid",
        "musicbrainz_releasegroupid",
        "musicbrainz_releasetrackid"
    ]
    with open("/Volumes/Data/acousticbrainz/data_collection.jsonl", "w") as f_out:
        for file in tqdm(files):
            with open(file) as f:
                content = json.load(f)

            relevant_contents = {
                "metadata": {},
                "moods": {}
            }

            for label in musicbrainz_labels:
                metadata = content.get("metadata", None)
                if metadata is not None:
                    tags = metadata.get("tags", None)
                    if tags is not None:
                        label_value = tags.get(label, None)
                    else:
                        label_value = None
                else:
                    label_value = None

                if label_value:
                    label_value = label_value[0]
                    if label == "musicbrainz_albumid":
                        if not is_valid_uuid(label_value):
                            label_value = None
                relevant_contents["metadata"][label] = label_value

            for mood in mood_labels:
                highlevel = content.get("highlevel", None)
                if highlevel is not None:
                    mood_data = highlevel.get(mood, None)
                    if mood_data is not None:
                        mood_value = mood_data.get("probability", None)
                    else:
                        mood_value = None
                else:
                    mood_value = None

                relevant_contents["moods"][mood.split("_")[1]] = mood_value

            f_out.write(json.dumps({file: relevant_contents}) + "\n")


def merge_moods():
    file_path = "/Volumes/Data/acousticbrainz/data_collection.jsonl"
    file_size = os.path.getsize(file_path)
    with open(file_path) as f:
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)
        merged_data = {}
        for line in f:
            progress_bar.update(len(line.encode('utf-8')))
            content = json.loads(line)
            for key in content:
                if all(value is not None for value in content[key]["moods"].values()):
                    album_id = content[key]["metadata"]["musicbrainz_albumid"]
                    artist_id = content[key]["metadata"]["musicbrainz_artistid"]
                    if album_id and is_valid_uuid(album_id):
                        if album_id in merged_data:
                            for mood in content[key]["moods"]:
                                merged_data[album_id][mood].append(content[key]["moods"][mood])
                        else:
                            data_dict = {"artist": artist_id}
                            for mood in content[key]["moods"]:
                                data_dict[mood] = [content[key]["moods"][mood]]
                            merged_data[album_id] = data_dict
        progress_bar.close()
    for key in tqdm(merged_data, desc="Calculating mean values"):
        for inner_key in merged_data[key]:
            if inner_key == "artist":
                continue
            mean_value = sum(merged_data[key][inner_key]) / len(merged_data[key][inner_key])
            merged_data[key][inner_key] = mean_value
    with open("/Volumes/Data/acousticbrainz/merged_data.jsonl", "w") as f:
        for key in tqdm(merged_data, desc="Saving merged data"):
            f.write(json.dumps({key: merged_data[key]}) + "\n")


def combine_datasets():
    covers_path = "/Volumes/Data/musicbrainz/covers"
    all_covers = [file for file in os.listdir(covers_path) if os.path.isfile(os.path.join(covers_path, file)) and "DS_Store" not in file]
    all_covers = [file.split(".")[0] for file in all_covers]

    all_files = "/Volumes/Data/acousticbrainz/merged_data.jsonl"
    file_size = os.path.getsize(all_files)

    combined_content = {}
    with open(all_files) as f:
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)
        for line in f:
            progress_bar.update(len(line.encode('utf-8')))
            content = json.loads(line)
            for key in content:
                if key in all_covers:
                    combined_content[key] = content[key]
        progress_bar.close()

    with open("/Volumes/Data/bothbrainz/combined_data.json", "w") as f:
        json.dump(combined_content, f)


def extract_genre_data():
    def count_file_lines(filename):
        print("Counting lines...")
        with open(filename, 'r') as f:
            return sum(1 for _ in f)
    collection = {}
    file_path = f"/Volumes/Data/musicbrainz/artist/mbdump/artist"

    total_lines = count_file_lines(file_path)
    with open(file_path) as artist_files:
        for artist in tqdm(artist_files, total=total_lines, desc="Processing artists"):
            content = json.loads(artist)
            if content["genres"]:
                collection[content["id"]] = content["genres"]
    with open("/Volumes/Data/bothbrainz/artist_genres.json", "w") as f:
        json.dump(collection, f)


def add_genre_data():
    genre_collection = {}
    file = "/Volumes/Data/bothbrainz/combined_data.json"
    genre_file = "/Volumes/Data/bothbrainz/artist_genres.json"
    with open(genre_file) as genre_data:
        genre_files = json.load(genre_data)
    with open(file) as data_file:
        for line in data_file:
            content = json.loads(line)
            for key, value in tqdm(content.items()):
                for artist_key, genre_list in genre_files.items():
                    if value["artist"] == artist_key:
                        value["genres"] = genre_list
                        genre_collection[key] = value
                        break
    with open("/Volumes/Data/bothbrainz/combined_data_with_genres.json", "w") as f:
        json.dump(genre_collection, f)


def process_genre_data():
    collection = {}
    with open("/Users/nico/code/CulturalAnalytics-CoverPredictions/data/genres.json") as f:
        genres = json.load(f)
    with open("/Volumes/Data/bothbrainz/combined_data_with_genres.json") as f:
        content = json.load(f)

    missing_genre = {
        "bf9955d6-c3eb-466d-95ca-30f3848fc9ea": {
            "genre": "bf9955d6-c3eb-466d-95ca-30f3848fc9ea",
            "name": "western classical",
            "subgenre of": [{"title": "classical", "id": "6ed4e4d1-9a97-4e2c-b8df-083754f154f4"}],
            "subgenres": [],
            "has fusion genres": [],
            "fusion of": [],
            "influenced by": [],
            "influenced genres": []
        }
    }
    genres.update(missing_genre)

    for key, value in tqdm(content.items()):
        sorted_genres = sorted(value["genres"], key=lambda x: x['count'], reverse=True)
        sorted_genres = [genre for genre in sorted_genres if genre["count"] == sorted_genres[0]["count"]]
        for genre in sorted_genres:
            if genre["id"] in genres:
                if genres[genre["id"]]["subgenre of"]:
                    value["genre_name"] = genre["name"]
                    value["genre"] = genre["id"]
                    del value["genres"]
                    collection[key] = value
                    break

    with open("/Volumes/Data/bothbrainz/finalized_data.json", "w") as f:
        json.dump(collection, f)


def plot_data():
    def find_top_level_genre_id(genre_id, genres):
        """Recursively find the top-level genre id for a given subgenre id."""
        parent_genre = genres.get(genre_id, {}).get("subgenre of")
        if parent_genre:
            # Recursively trace back to the top-level genre
            return find_top_level_genre_id(parent_genre[0]["id"], genres)
        else:
            # Top-level genre found
            return genre_id

    with open("/Users/nico/code/CulturalAnalytics-CoverPredictions/data/genres.json") as f:
        genres = json.load(f)
    with open("/Volumes/Data/bothbrainz/finalized_data.json") as f:
        content = json.load(f)

    missing_genre = {
        "bf9955d6-c3eb-466d-95ca-30f3848fc9ea": {
            "genre": "bf9955d6-c3eb-466d-95ca-30f3848fc9ea",
            "name": "western classical",
            "subgenre of": [{"title": "classical", "id": "6ed4e4d1-9a97-4e2c-b8df-083754f154f4"}],
            "subgenres": [],
            "has fusion genres": [],
            "fusion of": [],
            "influenced by": [],
            "influenced genres": []
        }
    }
    genres.update(missing_genre)

    subgenres = []
    for key, value in tqdm(content.items()):
        subgenres.append(value["genre"])

    subgenre_counts = Counter(subgenres)
    subgenre_counts = {item: count for item, count in subgenre_counts.items() if count >= 1000}

    subgenre_to_name = {genre_key: genre_val["genre"] for genre_key, genre_val in genres.items()}
    subgenre_to_top_level_genre = {}

    for sub_key, sub_val in subgenre_counts.items():
        for genre_key, genre_val in genres.items():
            if genre_val.get("subgenre of") and genre_val["genre"] == sub_key:
                top_level_genre_id = find_top_level_genre_id(genre_key, genres)
                subgenre_to_top_level_genre[genre_key] = top_level_genre_id

    relevant_genres = []
    top_level_genre_to_subgenres = {}
    for subgenre_id, top_level_genre_id in subgenre_to_top_level_genre.items():
        if subgenre_counts.get(subgenre_id, 0) >= 1000:
            if top_level_genre_id not in top_level_genre_to_subgenres:
                top_level_genre_to_subgenres[top_level_genre_id] = []
            top_level_genre_to_subgenres[top_level_genre_id].append(subgenre_id)

    for top_level_genre_id, subgenre_ids in top_level_genre_to_subgenres.items():
        labels = [subgenre_to_name[subgenre_id] + "|{}".format(subgenre_counts[subgenre_id]) for subgenre_id in
                  subgenre_ids]
        save_labels = [label.split("|")[0] for label in labels]

        dataset_content = {genres[top_level_genre_id]["genre"]: save_labels}
        relevant_genres.append(dataset_content)

        values = [subgenre_counts[subgenre_id] for subgenre_id in subgenre_ids]

        colors = [plt.cm.Spectral(i / len(labels)) for i in range(len(labels))]  # Generating a color for each subgenre

        # Update labels with genre names from the genres dict
        labels = [f"{genres[label.split('|')[0]]['name']}|{label.split('|')[1]}" for label in labels]

        plt.figure(figsize=(12, 8))
        squarify.plot(sizes=values, label=labels, color=colors, alpha=0.8)

        top_level_genre_name = genres[top_level_genre_id]["name"]
        #plt.title(f'Treemap of {top_level_genre_name}')
        plt.axis('off')

        plt.savefig(f"data/genre_treemap_{top_level_genre_name}.pdf")
        plt.close()
    with open("data/relevant_genres.json", "w") as f:
        json.dump(relevant_genres, f)


def sample_dataset():
    with open("/Volumes/Data/bothbrainz/finalized_data.json") as f:
        content = json.load(f)
    with open("data/relevant_genres.json") as f:
        genres = json.load(f)

    sampled_dataset = []
    for genre in genres:
        for k, subgenres in genre.items():
            if len(subgenres) > 5:
                selected_sample = random.sample(subgenres, 6)
                for subgenre in selected_sample:
                    possible_items = []
                    for key, value in content.items():
                        if value["genre"] == subgenre:
                            entry_dict = {
                                "metadata": {
                                    "album": key,
                                    "subgenre": value["genre"],
                                    "subgenre_name": value["genre_name"]
                                },
                                "moods": {
                                    "acoustic": value["acoustic"],
                                    "aggressive": value["aggressive"],
                                    "electronic": value["electronic"],
                                    "happy": value["happy"],
                                    "party": value["party"],
                                    "relaxed": value["relaxed"],
                                    "sad": value["sad"]
                                }
                            }
                            possible_items.append(entry_dict)
                    if len(possible_items) < 1000:
                        raise ValueError(f"Subgenre {subgenre} has less than 1000 entries")
                    selected_items = random.sample(possible_items, 1000)
                    for item in selected_items:
                        sampled_dataset.append(item)
    with open("data/finished_dataset.json", "w") as f:
        json.dump(sampled_dataset, f)


def save_genre_mapping():
    def find_top_level_genre_id(genre_id, genres):
        """Recursively find the top-level genre id for a given subgenre id."""
        parent_genre = genres.get(genre_id, {}).get("subgenre of")
        if parent_genre:
            # Recursively trace back to the top-level genre
            return find_top_level_genre_id(parent_genre[0]["id"], genres)
        else:
            # Top-level genre found
            return genre_id

    with open("data/finished_dataset.json") as f:
        content = json.load(f)
    with open("/Users/nico/code/CulturalAnalytics-CoverPredictions/data/genres.json") as f:
        genres = json.load(f)

    mapping = {}
    for item in content:
        subgenre_id = item["metadata"]["subgenre"]
        top_level_genre_id = find_top_level_genre_id(subgenre_id, genres)
        #mapping[subgenre_id] = top_level_genre_id

        subgenre_label = genres[subgenre_id]["name"]
        genre_label = genres[top_level_genre_id]["name"]
        mapping[subgenre_label] = genre_label

    with open("data/genre_mapping.json", "w") as f:
        json.dump(mapping, f)





#merge_moods()
#time.sleep(2)
#combine_datasets()
#extract_genre_data()
#add_genre_data()
#process_genre_data()
#plot_data()
sample_dataset()
save_genre_mapping()
