#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

"""
For each podcast episode:
* Download the raw mp3/m4a file
* Convert it to a 16k mono wav file
# Remove the original file
"""

import os
import pathlib
import subprocess
import sys
import shutil

import numpy as np
import requests
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Download raw audio files for SEP-28k or FluencyBank and convert to 16k hz mono wavs.')
parser.add_argument('--episodes', type=str, required=True,
                   help='Path to the labels csv files (e.g., SEP-28k_episodes.csv)')
parser.add_argument('--wavs', type=str, default="wavs",
                   help='Path where audio files from download_audio.py are saved')


args = parser.parse_args()
episode_uri = args.episodes
wav_dir = args.wavs

# Check if ffmpeg is available
if not shutil.which('ffmpeg'):
	print("ERROR: ffmpeg is not installed or not in PATH.")
	print("")
	print("Please install ffmpeg:")
	print("  Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH")
	print("           Or use: winget install ffmpeg")
	print("  Mac: brew install ffmpeg")
	print("  Linux: sudo apt install ffmpeg")
	sys.exit(1)

print(f"Using ffmpeg: {shutil.which('ffmpeg')}")
print()

# Load episode data
table = np.loadtxt(episode_uri, dtype=str, delimiter=",")
# Strip whitespace from all entries
table = np.char.strip(table)
urls = table[:,2]
n_items = len(urls)

audio_types = [".mp3", ".m4a", ".mp4"]


for i in range(n_items):
	# Get show/episode IDs
	show_abrev = table[i,-2]
	ep_idx = table[i,-1]
	episode_url = table[i,2]

	# Check file extension
	ext = ''
	for ext in audio_types:
		if ext in episode_url:
			break

	# Ensure the base folder exists for this episode
	episode_dir = pathlib.Path(f"{wav_dir}/{show_abrev}/")
	os.makedirs(episode_dir, exist_ok=True)

	# Get file paths
	audio_path_orig = pathlib.Path(f"{episode_dir}/{ep_idx}{ext}")
	wav_path = pathlib.Path(f"{episode_dir}/{ep_idx}.wav")

	# Check if this file has already been downloaded
	if os.path.exists(wav_path):
		continue

	print(f"Processing {i+1}/{n_items}: {show_abrev} {ep_idx}")
	
	try:
		# Download raw audio file
		if not os.path.exists(audio_path_orig):
			print(f"  Downloading from {episode_url}")
			response = requests.get(episode_url, stream=True, timeout=60)
			response.raise_for_status()
			
			total_size = int(response.headers.get('content-length', 0))
			with open(audio_path_orig, 'wb') as f:
				for chunk in tqdm(response.iter_content(chunk_size=8192), 
								  total=total_size//8192, 
								  unit='KB', 
								  desc="  Downloading",
								  leave=False):
					f.write(chunk)
			print("  Download complete")
		
		# Convert to 16khz mono wav file
		print("  Converting to 16kHz mono WAV...")
		ffmpeg_cmd = [
			'ffmpeg',
			'-i', str(audio_path_orig),
			'-ac', '1',  # mono
			'-ar', '16000',  # 16kHz sample rate
			'-y',  # overwrite output file
			'-loglevel', 'error',  # only show errors
			str(wav_path)
		]
		result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
		if result.returncode != 0:
			raise Exception(f"ffmpeg conversion failed: {result.stderr}")
		print("  Conversion complete")
		
		# Remove the original mp3/m4a file
		if os.path.exists(audio_path_orig):
			os.remove(audio_path_orig)
			print("  Cleaned up original file")
			
	except Exception as e:
		print(f"  ERROR: Failed to process {show_abrev} {ep_idx}: {str(e)}")
		# Clean up partial files
		if os.path.exists(audio_path_orig):
			os.remove(audio_path_orig)
		continue
