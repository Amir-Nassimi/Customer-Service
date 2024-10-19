import os
from pydub import AudioSegment

class AudioChunkProcessor:
    def __init__(self, diarizer, chunk_duration_sec=10):
        self.diarizer = diarizer
        self.chunk_duration_ms = chunk_duration_sec * 1000  # Convert to milliseconds

    def process_chunks(self, audio_file, emotion_obj, volume_obj):
        # Load the audio file using pydub
        audio = AudioSegment.from_file(audio_file)
        total_length_ms = len(audio)

        # Split into chunks and analyze
        all_results = []

        for start_ms in range(0, total_length_ms, self.chunk_duration_ms):
            end_ms = min(start_ms + self.chunk_duration_ms, total_length_ms)
            chunk = audio[start_ms:end_ms]

            # Save the chunk to a temporary file
            chunk_file = f"chunk_{start_ms//1000}_{end_ms//1000}.wav"
            chunk.export(chunk_file, format="wav")

            # Process the chunk with the diarizer
            results = self.diarizer.process_audio(chunk_file, emotion_obj, volume_obj)

            # Adjust the results with the actual starting point
            for result in results:
                result["start"] += start_ms / 1000
                result["stop"] += start_ms / 1000
            
            all_results.extend(results)

            # Remove the temporary file to save space
            os.remove(chunk_file)

        return all_results