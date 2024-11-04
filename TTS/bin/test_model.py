import os
import torch
import json
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

if __name__ == "__main__":
    # Define paths
    output_path = "D:/output2/run-October-26-2024_05+01PM-dbf1a08a"
    model_path = os.path.join(output_path, "best_model.pth")
    config_path = os.path.join(output_path, "config.json")

    # Load and filter configuration
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Get the valid attributes for GlowTTSConfig
    valid_keys = GlowTTSConfig.__annotations__.keys()
    filtered_config_data = {k: v for k, v in config_data.items() if k in valid_keys}
    
    # Initialize GlowTTSConfig with filtered data
    config = GlowTTSConfig(**filtered_config_data)

    # Ensure characters config is an object, not a dictionary
    if isinstance(config.characters, dict):
        from TTS.tts.utils.text.characters import CharactersConfig
        config.characters = CharactersConfig(**config.characters)

    # Initialize Audio Processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize Tokenizer
    tokenizer, _ = TTSTokenizer.init_from_config(config)

    # Load the trained model
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # Load the model state, allowing missing keys in state_dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)
    model.eval()

    # Text to synthesize
    text = "Hello, this is a test of the text-to-speech model."

    # Tokenize input text
    tokens = tokenizer.text_to_ids(text)  # Use text_to_ids instead of text_to_sequence
    
    # Convert tokens to a tensor
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    x_lengths = torch.tensor([tokens_tensor.size(1)], dtype=torch.long).unsqueeze(0)  # Create lengths tensor with shape [1, 1]

    # Generate audio
    with torch.no_grad():
        audio, _ = model.inference(tokens_tensor, x_lengths)  # Pass both tokens and lengths

    # Save the audio
    output_audio_path = os.path.join(output_path, "output.wav")
    ap.save_wav(audio, output_audio_path)

    print(f"Audio saved to {output_audio_path}")