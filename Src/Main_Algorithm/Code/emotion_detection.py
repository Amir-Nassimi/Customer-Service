import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2FeatureExtractor

import os
import sys
import numpy as np
from io import BytesIO
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[2]))
from utils.models import Wav2Vec2ForSpeechClassification


class EmotionDetection:
    def __init__(self, model_name = 'm3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition'):
        self.config = AutoConfig.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name).to(self.device)
    
    def speech_file_to_array_fm(self, cropped_audio, sampling_rate):
        audio_bytes = BytesIO()
        cropped_audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)  # Reset the stream position to the beginning

        speech_array, _sampling_rate = torchaudio.load(audio_bytes)
        resampler = torchaudio.transforms.Resample(sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def process_audio_segment(self, path):
        speech = self.speech_file_to_array_fm(path, self.sampling_rate)
        inputs = self.feature_extractor(speech, sampling_rate=self.sampling_rate, return_tensors='pt', padding=True)
        inputs = {key:inputs[key].to(self.device) for key in inputs}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        i = np.argmax(scores)
        outputs = f'{self.config.id2label[i]}: {round(scores[i]*100, 3):.2f}%'
        # outputs = [{"Label": self.config.id2label[i], "Score":f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
        return outputs