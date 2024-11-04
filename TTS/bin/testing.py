import glob
import os
import subprocess
import IPython.display as display
from pydub import AudioSegment
from pydub.playback import play


# Define output path
output_path = "D:/output2/run-October-26-2024_05+01PM-dbf1a08a"

# Gather the sorted list of checkpoint and configuration files
ckpts = sorted([f for f in glob.glob(os.path.join(output_path, "best_model.pth"))])
configs = sorted([f for f in glob.glob(os.path.join(output_path, "config.json"))])

# Check if we have at least one checkpoint and one config
if not ckpts or not configs:
    print("No checkpoints or configs found.")
else:
    # Select the latest checkpoint and config (you can customize this logic)
    test_ckpt = ckpts[-1]  # Get the latest checkpoint
    test_config = configs[-1]  # Get the latest config

    custom_output_filename = "output.wav"  # Change this as needed
    custom_output_path = os.path.join("D:/output2/run-October-26-2024_05+01PM-dbf1a08a", custom_output_filename)

    os.makedirs(os.path.dirname(custom_output_path), exist_ok=True)

    # Prepare the command to run TTS
    command = [
        'tts',  # Assuming 'tts' is the command for your TTS tool
        '--text', 'hello my name is ishank goel and i am a very good student',
        '--model_path', test_ckpt,
        '--config_path', test_config,
        '--out_path', custom_output_path
    ]

    # Run the TTS command
    subprocess.run(command)

    # Play the output audio
    display.Audio(custom_output_path)

