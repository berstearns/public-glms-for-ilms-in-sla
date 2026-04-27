from pipeline._cli import run_stage

from .config import SECTION_MAP, TrainIlmConfig
from .runner import run_train_ilm


if __name__ == "__main__":
    run_stage(
        description="Continued-pretrain a GLM on EFCAMDAT with GLM objective.",
        config_cls=TrainIlmConfig,
        section_map=SECTION_MAP,
        runner=run_train_ilm,
        use_run_dir=False,
    )
