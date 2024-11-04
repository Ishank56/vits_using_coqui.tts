if __name__ == "__main__":
    import os

    # Trainer: Where the ✨️ happens.
    # TrainingArgs: Defines the set of arguments of the Trainer.
    from trainer import Trainer, TrainerArgs

    # GlowTTSConfig: all model related values for training, validating and testing.
    from TTS.tts.configs.glow_tts_config import GlowTTSConfig

    # BaseDatasetConfig: defines name, formatter and path of the dataset.
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.glow_tts import GlowTTS
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    # Set the output path directly to D:/output1
    output_path = "D:/output2"

    # DEFINE DATASET CONFIG
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "D:/hindi_tts_dataset")
    )

    # INITIALIZE THE TRAINING CONFIGURATION
    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=100,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
    )

    # INITIALIZE THE AUDIO PROCESSOR
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # INITIALIZE THE MODEL
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # INITIALIZE THE TRAINER
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # AND... 3,2,1... 🚀
    trainer.fit()
