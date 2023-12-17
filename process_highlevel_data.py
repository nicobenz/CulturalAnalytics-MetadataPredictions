import json
import os
from icecream import ic

data_path = "data/highlevel_data"
"""
# get the path of all files in the nested directories
# maybe better readability when not using list comprehension?
record_ids = [
    [
        [
            f"{data_path}/{first_dir[0:2]}/{second_dir[0:1]}{rec_id}" for rec_id in os.listdir(f"{data_path}/{first_dir}/{second_dir}") if rec_id != ".DS_Store"
        ] for second_dir in os.listdir(f"{data_path}/{first_dir}") if second_dir != ".DS_Store"
    ] for first_dir in os.listdir(data_path) if first_dir != ".DS_Store"
]
"""
record_ids = []
for first_level in os.listdir(data_path):
    if first_level != ".DS_Store":
        deeper = f"{data_path}/{first_level}"
        for second_level in os.listdir(deeper):
            even_deeper = f"{data_path}/{first_level}/{second_level}"
            if second_level != ".DS_Store":
                for rec_id in os.listdir(even_deeper):
                    rec_id_path = f"{data_path}/{first_level}/{second_level}/{rec_id}"
                    record_ids.append(rec_id_path)

count = 0
mapped_ids = {}
for rec_id in record_ids:
    with open(rec_id, "r") as f:
        recording = json.load(f)
    if recording is not None and "metadata" in recording and recording["metadata"] is not None and "tags" in recording[
        "metadata"] and recording["metadata"]["tags"] is not None and "musicbrainz_releasegroupid" in \
            recording["metadata"]["tags"]:
        release_id = recording["metadata"]["tags"]["musicbrainz_releasegroupid"]
        for rel_id in release_id:
            if rel_id in mapped_ids:
                mapped_ids[rel_id].append(rec_id)
            else:
                mapped_ids[rel_id] = [rec_id]
    else:
        count += 1

print(count, len(record_ids))
