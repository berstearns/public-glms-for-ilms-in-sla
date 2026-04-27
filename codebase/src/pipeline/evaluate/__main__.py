from pipeline._cli import run_stage

from .config import SECTION_MAP, EvaluateConfig
from .runner import run_evaluate


if __name__ == "__main__":
    run_stage(
        description="Score predictions; emit summary + stratified reports.",
        config_cls=EvaluateConfig,
        section_map=SECTION_MAP,
        runner=run_evaluate,
        use_run_dir=False,
    )
