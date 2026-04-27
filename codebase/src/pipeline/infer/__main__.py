from pipeline._cli import run_stage

from .config import SECTION_MAP, InferStageConfig
from .runner import run_infer


if __name__ == "__main__":
    run_stage(
        description="Run the configured backbone on the cloze JSONL.",
        config_cls=InferStageConfig,
        section_map=SECTION_MAP,
        runner=run_infer,
        use_run_dir=False,
    )
