

from pathlib import Path
from auto_video_labeller import run
import os

source=r"D:\Teknofest\YOLOVA\Veriseti\video\KesisenYol2.mp4"
run(source=source, output_folder=str(Path(source).parent))
