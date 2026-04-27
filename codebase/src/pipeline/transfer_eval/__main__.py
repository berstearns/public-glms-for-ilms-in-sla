from pipeline._cli import run_stage

from .config import SECTION_MAP, TransferEvalConfig
from .runner import run_transfer_eval


if __name__ == "__main__":
    run_stage(
        description="Apply the trained checkpoint to transfer corpora.",
        config_cls=TransferEvalConfig,
        section_map=SECTION_MAP,
        runner=run_transfer_eval,
        use_run_dir=False,
    )
