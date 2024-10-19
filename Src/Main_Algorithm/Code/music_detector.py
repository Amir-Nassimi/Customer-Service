import os
import sys
import torch
import ffmpeg
import librosa
import tempfile
from pathlib import Path
from singleton_decorator import singleton

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[2]))
from utils.samplecnn import SampleCNN

@singleton
class Music_Detection:
    def __init__(self, model_pth=f'{os.path.abspath(Path(__file__).resolve().parents[3])}/Models/model-gztan-speech-music-20000.pth'):
        # self.MICRO = "\U0001F3A4"
        # self.MUSIC = "\U0001F3B6"
        self.thrsh = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_pth)

    def _load_model(self, model_pth):
        self.net = SampleCNN().to(self.device)
        self.net.load_state_dict(torch.load(model_pth))
        self.net.eval()
    
    @staticmethod
    def strfdelta(t):
        d = {}
        d["h"], rem = divmod(t, 3600)
        d["m"], d["s"] = divmod(rem, 60)
        return d

    def analyze(self, file_pth):
        music_value = []
        with tempfile.TemporaryDirectory(suffix="-discriminator") as tmpdir:
            # Convert input to proper format (16 kHz, mono)
            output = os.path.join(tmpdir, "16khz.wav")
            (
                ffmpeg.input(file_pth)
                    .output(output, format="wav", acodec="pcm_s16le", ac=1, ar="16k")
                    .overwrite_output()
                    .run()
            )

            with torch.no_grad():
                for i, frame in enumerate(map(torch.Tensor, librosa.stream(output, block_length=1, frame_length=59049, hop_length=16000, fill_value=0))):
                    y = self.net(frame.reshape(1, -1).to(self.device))
                    # print("{h:02d}:{m:02d}:{s:02d}".format(**self.strfdelta(i)), "{} ({:.3f})".format(self.MICRO if y > self.thrsh else self.MUSIC, y))
                    
                    if y.item() <= self.thrsh:
                        music_value.append("{h:02d}:{m:02d}:{s:02d}".format(**self.strfdelta(i)))

        return music_value