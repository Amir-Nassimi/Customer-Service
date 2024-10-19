from singleton_decorator import singleton

@singleton
class Handling:
    @staticmethod
    def analyze_hold(time_stamps):
        def extract_time(ts):
            return ts.split()[0]

        # Convert time stamps to a list of times
        times = [extract_time(ts) for ts in time_stamps]

        # Identify continuous sections
        sections = []
        try:
            start = times[0]
        except IndexError:
            return ''

        # Loop through the list to detect breaks and record sections
        for i in range(1, len(times)):
            current_time = times[i]
            previous_time = times[i - 1]
            
            # Calculate the difference in seconds
            time_gap = int(current_time[-2:]) - int(previous_time[-2:])
            
            if time_gap > 1:  # Break in the sequence
                sections.append((start, previous_time))
                start = current_time

        # Add the final section
        sections.append((start, times[-1]))

        # Generate human-readable messages for each section
        output_messages = [f"Hold from {s[0]} to {s[1]}" for s in sections]

        return output_messages

    def analyze_results(self, results, hold_result):
        speaker_count = {}
        speaker_duration = {}
        total_silence = 0
        previous_end = 0
        previous_speaker = None

        for result in results:
            start = result["start"]
            stop = result["stop"]
            speaker = result["speaker"]

            # Count the transitions (interruptions)
            if speaker != previous_speaker:
                if speaker not in speaker_count:
                    speaker_count[speaker] = 0
                    speaker_duration[speaker] = 0
                speaker_count[speaker] += 1  # New turn

            # Calculate the total duration for each speaker in milliseconds
            duration = (stop - start) * 1000  # convert to milliseconds
            speaker_duration[speaker] += duration

            # Calculate the silence between segments
            if start > previous_end:
                silence = (start - previous_end) * 1000  # convert to milliseconds
                total_silence += silence

            previous_end = stop
            previous_speaker = speaker  # Track the last speaker

        total_duration = previous_end * 1000  # total length in milliseconds

        return {
            "speaker_count": speaker_count,
            "speaker_duration": speaker_duration,
            "total_silence": total_silence,
            "total_duration": total_duration,
            "hold_duration": self.analyze_hold(hold_result)
        }
    