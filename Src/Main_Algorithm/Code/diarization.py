import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline


class SpeakerDiarization:
    def __init__(self, model_name: str, access_token: str):
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=access_token
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)

    def process_audio(self, audio_file: str, emotion_detector, volume_detector):
        diarization = self.pipeline(audio_file)

        # Load the entire audio
        audio = AudioSegment.from_file(audio_file)

        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            stop = turn.end
            # Ignore segments shorter than 0.5 seconds
            if (stop - start) < 0.5:
                continue

            # Crop the audio segment for this turn
            segment_start = int(start * 1000)  # convert to milliseconds
            segment_stop = int(stop * 1000)  # convert to milliseconds
            cropped_audio = audio[segment_start:segment_stop]
     
            volume_out = volume_detector.analyze(cropped_audio)
            emotion_out = emotion_detector.process_audio_segment(cropped_audio)

            result = {
                "start": start,
                "stop": stop,
                "speaker": f"speaker_{speaker}",
                "emotion": emotion_out,
                "Average Volume (dBFS)": volume_out
            }
            results.append(result)

        return results
