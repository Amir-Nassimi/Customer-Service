# Custumer Service
In this program a novel approach is designed in order to handle the Custumer Services. The major points in this project is as follow:
1. Emotion Detection
2. Speech Diarization
3. 'Hold' Detection during Speech
4. Counting the number of talking of each speaker
5. Counting the amount of being silence in the entire speech
6. Analyzing the Volume level of each speaker

# Installation
1. Install pytorch utilizing the following command:
```shell script
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Install dependencies via `requirements.txt`:
```shell script
pip install -r requirements.txt
```

# Usage
1. Import important libraries from the utils file:

```python
from Src.utils.handler import Handling
from Src.utils.music_detector import Music_Detection
from Src.utils.diarization import SpeakerDiarization
from Src.utils.audio_chunk import AudioChunkProcessor
from Src.utils.emotion_detection import EmotionDetection
from Src.utils.volume_analysis import VoiceVolumeAnalyzer
```

2. Run the followings:

```python
    use_chunks = False  # Whether to process the audio in chunks
    audio_file = 'Path to audio file'

    diarizer = SpeakerDiarization("pyannote/speaker-diarization-3.1", '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')

    if use_chunks:
        # Process the audio in chunks
        chunk_processor = AudioChunkProcessor(diarizer)
        results = chunk_processor.process_chunks(audio_file, EmotionDetection(), VoiceVolumeAnalyzer())
    else:
        # Process the entire audio file at once
        results = diarizer.process_audio(audio_file, EmotionDetection(), VoiceVolumeAnalyzer())

    music_result = Music_Detection().analyze(audio_file)
    analysis = Handling().analyze_results(results, music_result)
```

3. Print the Outputs:

```python
    print("Diarization Results:")
    for result in results:
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

                start_seconds = time_to_seconds(start_time)
                end_seconds = time_to_seconds(end_time)

                duration_seconds += (end_seconds - start_seconds)

        print(f"\nTotal Hold: {duration_seconds} s\n")
```

# Prediction
Run the ocr via the following command:
```shell script
    python ./test/test.py
```

# Output
The output of this model would be as follow:

Diarization Results:
start: 0.8s, stop: 2.1s, Speaker: speaker_SPEAKER_00, Emotion: Neutral: 61.95%, Average Volume (dBFS): -14.02 is Normal

Speaker Count:
Speaker speaker_SPEAKER_00: 1 times

Speaker Duration (Seconds):
Speaker speaker_SPEAKER_00: 1.26 s

Total Silence: 0.84 s
Total Duration: 2.10 s
Total Hold: None Holding

or

start: 299.9s, stop: 304.9s, Speaker: speaker_SPEAKER_01, Emotion: Anger: 99.96%, Average Volume (dBFS): -17.48 is Normal
start: 304.9s, stop: 305.8s, Speaker: speaker_SPEAKER_02, Emotion: Anger: 99.83%, Average Volume (dBFS): -19.85 is Normal
start: 315.5s, stop: 317.3s, Speaker: speaker_SPEAKER_01, Emotion: Anger: 99.84%, Average Volume (dBFS): -19.26 is Normal
start: 337.3s, stop: 338.1s, Speaker: speaker_SPEAKER_00, Emotion: Anger: 99.94%, Average Volume (dBFS): -28.52 is Unusual
start: 338.8s, stop: 345.0s, Speaker: speaker_SPEAKER_00, Emotion: Anger: 99.95%, Average Volume (dBFS): -30.29 is Unusual
start: 344.9s, stop: 347.4s, Speaker: speaker_SPEAKER_01, Emotion: Anger: 99.95%, Average Volume (dBFS): -21.08 is Unusual
start: 346.6s, stop: 347.1s, Speaker: speaker_SPEAKER_00, Emotion: Anger: 99.89%, Average Volume (dBFS): -24.52 is Unusual  
start: 347.5s, stop: 348.8s, Speaker: speaker_SPEAKER_00, Emotion: Anger: 99.92%, Average Volume (dBFS): -31.50 is Unusual 
start: 349.1s, stop: 351.6s, Speaker: speaker_SPEAKER_01, Emotion: Anger: 99.95%, Average Volume (dBFS): -21.79 is Unusual  
start: 353.1s, stop: 356.0s, Speaker: speaker_SPEAKER_02, Emotion: Anger: 99.95%, Average Volume (dBFS): -19.03 is Normal   
start: 356.0s, stop: 357.0s, Speaker: speaker_SPEAKER_01, Emotion: Anger: 98.30%, Average Volume (dBFS): -22.54 is Unusual  

Speaker Count:                                                                                                          
Speaker speaker_SPEAKER_01: 14 times                                                                                    
Speaker speaker_SPEAKER_02: 6 times                                                                                     
Speaker speaker_SPEAKER_00: 8 times

Speaker Duration (Seconds):                                                                                             
Speaker speaker_SPEAKER_01: 136.26 s                                                                                    
Speaker speaker_SPEAKER_02: 10.53 s                                                                                     
Speaker speaker_SPEAKER_00: 45.52 s

Total Silence: 180.35 s                                                                                                 
Total Duration: 357.00 s

Hold Duration:                                                                                                          
Hold from 00:01:43 to 00:05:35
Total Hold: 232 s 


# Metrics
1. Acc = 100%
2. Speed = 20.24 on a 5 minute audio file
3. Memory = 71.8%                                                                                                