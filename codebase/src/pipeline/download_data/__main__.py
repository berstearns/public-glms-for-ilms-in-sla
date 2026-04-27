from pipeline._cli import run_stage

from .config import SECTION_MAP, DownloadDataConfig
from .runner import run_download_data


if __name__ == "__main__":
    run_stage(
        description="Verify splits directory is accessible; emit an inventory.",
        config_cls=DownloadDataConfig,
        section_map=SECTION_MAP,
        runner=run_download_data,
        use_run_dir=False,
    )
