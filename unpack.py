import zstandard as zstd
import tarfile
import os
from tqdm import tqdm


def decompress_zstd(file_path, output_path):
    with open(file_path, 'rb') as compressed:
        with open(output_path, 'wb') as decompressed:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(compressed, decompressed)


def extract_zstd():
    path = '/Volumes/Data/acousticbrainz'
    files = [file for file in os.listdir(path) if file.endswith('.zst')]

    for file in tqdm(files):
        save_file = file.replace('.zst', '')
        input_file_path = f"{path}/{file}"
        output_file_path = f"{path}_unpacked/{save_file}"
        decompress_zstd(input_file_path, output_file_path)


def extract_tar():
    source_directory = '/Volumes/Data/acousticbrainz_unpacked'
    target_directory = '/Volumes/Data/acousticbrainz_untared'
    for filename in tqdm(os.listdir(source_directory)):
        if filename.endswith(".tar"):
            # Construct full file path
            file_path = os.path.join(source_directory, filename)

            # Open the tar file for reading
            with tarfile.open(file_path, 'r') as tar:
                # Extract all contents to target directory
                tar.extractall(path=target_directory)


extract_tar()
