import re
import os , sys
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[0]))
from handler import Handling
from music_detector import Music_Detection
from diarization import SpeakerDiarization
from audio_chunk import AudioChunkProcessor
from emotion_detection import EmotionDetection
from volume_analysis import VoiceVolumeAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Custumer Service")
    parser.add_argument("audio_file", help="Path to the input audio file")
    parser.add_argument(
        "access_token",
        nargs="?",
        default="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",
        help="Hugging Face access token for model authentication (default token provided)"
    )
    parser.add_argument(
        "--use-chunks",
        action="store_true",
        help="Enable processing of audio file in chunks"
    )
    return parser.parse_args()

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

def main():
    args = parse_args()
    diarizer = SpeakerDiarization("pyannote/speaker-diarization-3.1", args.access_token)

    if args.use_chunks:
        # Process the audio in chunks
        chunk_processor = AudioChunkProcessor(diarizer)
        results = chunk_processor.process_chunks(args.audio_file, EmotionDetection(), VoiceVolumeAnalyzer())
    else:
        # Process the entire audio file at once
        results = diarizer.process_audio(args.audio_file, EmotionDetection(), VoiceVolumeAnalyzer())

    music_result = Music_Detection().analyze(args.audio_file)

    analysis = Handling().analyze_results(results, music_result)

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
