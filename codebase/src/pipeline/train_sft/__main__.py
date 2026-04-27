from pipeline._cli import run_stage

from .config import SECTION_MAP, TrainSftConfig
from .runner import run_train_sft


if __name__ == "__main__":
    run_stage(
        description="Supervised fine-tuning on cloze triples.",
        config_cls=TrainSftConfig,
        section_map=SECTION_MAP,
        runner=run_train_sft,
        use_run_dir=False,
    )
