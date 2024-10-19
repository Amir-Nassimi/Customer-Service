from singleton_decorator import singleton

@singleton
class VoiceVolumeAnalyzer:

    @staticmethod
    def analyze(audio_segment):
        # Get the average dBFS and return it
        return audio_segment.dBFS
