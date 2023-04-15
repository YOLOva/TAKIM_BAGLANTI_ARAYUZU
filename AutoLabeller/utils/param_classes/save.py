

from tkinter import BooleanVar, IntVar


class SaveParams:

    def __init__(self, data=None) -> None:
        if data is None:
            return
        self.video_fps_to_save = IntVar(value=data["video_fps_to_save"])
        self.decrease_video_frame = BooleanVar(
            value=data["decrease_video_frame"])
        self.pass_detection_not_saved_frames = BooleanVar(
            value=data["pass_detection_not_saved_frames"])

    def toJson(self):
        return {
            "decrease_video_frame": self.decrease_video_frame.get(),
            "video_fps_to_save": self.video_fps_to_save.get(),
            "pass_detection_not_saved_frames": self.pass_detection_not_saved_frames.get()
        }
