import re
import psutil
import os , sys
from time import time
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from Code.handler import Handling
from Code.music_detector import Music_Detection
from Code.diarization import SpeakerDiarization
from Code.audio_chunk import AudioChunkProcessor
from Code.emotion_detection import EmotionDetection
from Code.volume_analysis import VoiceVolumeAnalyzer


def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

def main():
    use_chunks = False  # Whether to process the audio in chunks
    audio_file = f'{os.path.abspath(Path(__file__).resolve().parents[3])}/Dataset/test.wav'

    diarizer = SpeakerDiarization("pyannote/speaker-diarization-3.1", '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')

    x = time()

    if use_chunks:
        # Process the audio in chunks
        chunk_processor = AudioChunkProcessor(diarizer)
        results = chunk_processor.process_chunks(audio_file, EmotionDetection(), VoiceVolumeAnalyzer())
    else:
        # Process the entire audio file at once
        results = diarizer.process_audio(audio_file, EmotionDetection(), VoiceVolumeAnalyzer())

    music_result = Music_Detection().analyze(audio_file)

    analysis = Handling().analyze_results(results, music_result)

    print(f'\n\nTime: {time() - x}')
    print(f'Memory: {psutil.virtual_memory().percent}')

    # Output the results
    print("Diarization Results:")
    for result in results:
        # if result['speaker'] == 'speaker_SPEAKER_00':
        print(f"start: {result['start']:.1f}s, stop: {result['stop']:.1f}s, Speaker: {result['speaker']}, Emotion: {result['emotion']}, Average Volume (dBFS): {result['Average Volume (dBFS)']:.2f} is {'Normal' if (result['Average Volume (dBFS)']>= -20 and result['Average Volume (dBFS)']<= -10) else 'Unusual'}")

    print("\nSpeaker Count:")
    for speaker, count in analysis["speaker_count"].items():
        print(f"Speaker {speaker}: {count} times")

    print("\nSpeaker Duration (Seconds):")
    for speaker, duration in analysis["speaker_duration"].items():
        print(f"Speaker {speaker}: {duration/1000:.2f} s")

    print(f"\nTotal Silence: {analysis['total_silence']/1000:.2f} s")
    print(f"Total Duration: {analysis['total_duration']/1000:.2f} s")
    
    if analysis['hold_duration'] == '':
        print("Total Hold: None Holding")
    else:
        duration_seconds = 0
        print("\nHold Duration:")
        for hold in analysis['hold_duration']:
            print(hold)

            times = re.findall(r"\d{2}:\d{2}:\d{2}", hold)

            if len(times) == 2:
                start_time = times[0]
                end_time = times[1]

                # Convert start and end times to seconds
                start_seconds = time_to_seconds(start_time)
                end_seconds = time_to_seconds(end_time)

                # Calculate the duration in seconds
                duration_seconds += (end_seconds - start_seconds)

        print(f"\nTotal Hold: {duration_seconds} s\n")

if __name__ == "__main__":
    main()
