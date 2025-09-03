import os

from rich.console import Console

DEEP_ISLES_PAPER_URL = "https://pubmed.ncbi.nlm.nih.gov/40783484/"
DEEP_ISLES_PAPER_TEXT = 'de la Rosa, Ezequiel, et al. "DeepISLES: a clinically validated ischemic stroke segmentation model from the ISLES\'22 challenge." Nature Communications 16.1 (2025): 7357.'

NVAUTO_PAPER_URL = "https://arxiv.org/abs/2209.09546"
NVAUTO_PAPER_TEST = 'Siddique, Md Mahfuzur Rahman, et al. "Automated ischemic stroke lesion segmentation from 3D MRI." arXiv preprint arXiv:2209.09546 (2022).'

BRAINLES_PAPER_URL = "https://arxiv.org/abs/2507.09036"
BRAINLES_PAPER_TEXT = 'Kofler, Florian, et al. "BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis." arXiv preprint arXiv:2507.09036 (2025).'


def citation_reminder(func):
    """Decorator to remind users to cite stroke-segmentor."""

    def wrapper(*args, **kwargs):
        if (
            os.environ.get("STROKE_SEGMENTOR_CITATION_REMINDER", "true").lower()
            == "true"
        ):
            console = Console()
            console.rule("Thank you for using [bold]stroke-segmentor[/bold]")
            console.print(
                "This is the NVAUTO algorithmic solution from the ISLES'22 challenge.",
                justify="center",
            )
            console.print(
                "If you are using this tool, please cite the following work:",
                justify="center",
            )
            console.print(
                f"[italic]{NVAUTO_PAPER_TEST}[/italic] ({NVAUTO_PAPER_URL})",
                justify="center",
            )
            console.print(
                f"[italic]{DEEP_ISLES_PAPER_TEXT}[/italic] ({DEEP_ISLES_PAPER_URL})",
                justify="center",
            )
            console.print(
                f"[italic]{BRAINLES_PAPER_TEXT}[/italic] ({BRAINLES_PAPER_URL})",
                justify="center",
            )
            console.rule()
            console.line()
            os.environ["PANOPTICA_CITATION_REMINDER"] = "false"  # Show only once
        return func(*args, **kwargs)

    return wrapper
