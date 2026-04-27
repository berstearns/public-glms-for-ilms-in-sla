from pipeline._cli import run_stage

from .config import SECTION_MAP, PrepareSplitsConfig
from .runner import run_prepare_splits


if __name__ == "__main__":
    run_stage(
        description="CEFR-stratified sampling of EFCAMDAT train/test.",
        config_cls=PrepareSplitsConfig,
        section_map=SECTION_MAP,
        runner=run_prepare_splits,
        use_run_dir=False,
    )
