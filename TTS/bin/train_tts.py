import os
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs

from TTS.config import load_config, register_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model

import sys
print("Current sys.path:", sys.path)

@dataclass
class TrainTTSArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})

def main():
    """Run `tts` model training directly by a `config.json` file."""
    # init trainer args
    train_args = TrainTTSArgs()
    parser = train_args.init_argparse(arg_prefix="")

    # override trainer args from command-line args
    args, config_overrides = parser.parse_known_args()
    train_args.parse_args(args)

    # load config.json and register
    if args.config_path or args.continue_path:
        if args.config_path:
            # init from a file
            config = load_config(args.config_path)
            print("Loaded config from:", args.config_path)
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        elif args.continue_path:
            # continue from a previous experiment
            config = load_config(os.path.join(args.continue_path, "config.json"))
            print("Loaded config from continue path:", args.continue_path)
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        else:
            # init from console args
            from TTS.config.shared_configs import BaseTrainingConfig  # pylint: disable=import-outside-toplevel

            config_base = BaseTrainingConfig()
            config_base.parse_known_args(config_overrides)
            config = register_config(config_base.model)()
            print("Initialized config from console args")

    print("Config parameters:", config)  # Debug: print the loaded config parameters

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

    # init the model from config
    model = setup_model(config, train_samples + eval_samples)

    print("Model setup complete. Model details:", model)  # Debug: print model details

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    
    print("Trainer initialized. Starting training...")  # Debug: trainer initialization message
    trainer.fit()

if __name__ == "__main__":
    main()
