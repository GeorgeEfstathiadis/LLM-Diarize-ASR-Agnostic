import re
import json
import tgt

def true_labels(f):
    """Extract speaker labels and spoken words from Fisher transcript file."""
    with open(f, 'r') as file:
        transcript = file.read()
    # Initialize a dictionary to store speaker labels and their words
    ref_labels = []
    ref_words = []
    ref_times = []
    # Split the input text into lines and process each line
    for line in transcript.strip().split('\n'):
        # Extract the speaker label and the spoken words using regular expression
        match = re.match(r'(\d+\.\d+)\s+(\d+\.\d+)\s+([A-Z]):\s+(.*)', line)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            speaker_label = match.group(3)
            words = match.group(4)
            # Remove any annotations like laughter or actions
            words_cleaned = re.sub(r'\[.*?\]', '', words).strip().lower()
            # remove double (( and double ))
            words_cleaned = re.sub(r'[\(\)]', '', words_cleaned).strip()
            # remove _ and -
            words_cleaned = re.sub(r'[_-]', ' ', words_cleaned).strip()
            if not words_cleaned:
                continue
            # Split the cleaned words into a list
            words_list = words_cleaned.split()
            # Append the speaker label and the list of words to the dictionary
            ref_labels += [speaker_label] * len(words_list)
            ref_words += words_list
            ref_times += [(start_time, end_time)] * len(words_list)
    ref_labels = ['1' if x == 'A' else '2' for x in ref_labels]
    return ref_labels, ref_words, ref_times

def true_labels_primock57(directory, f):
    """Extract speaker labels and spoken words from PriMock57 dataset."""
    patient_grid = tgt.io.read_textgrid(f'{directory}/{f}_patient.TextGrid')
    doctor_grid = tgt.io.read_textgrid(f'{directory}/{f}_doctor.TextGrid')

    # for each speaker extract xmin, xmax, text
    speaker_labels = []
    speaker_intervals = []
    speaker_texts = []

    for i in range(len(patient_grid.tiers[0].intervals)):
        speaker_labels.append('patient')
        speaker_intervals.append((patient_grid.tiers[0].intervals[i].start_time, patient_grid.tiers[0].intervals[i].end_time))
        speaker_texts.append(patient_grid.tiers[0].intervals[i].text)

    for i in range(len(doctor_grid.tiers[0].intervals)):
        speaker_labels.append('doctor')
        speaker_intervals.append((doctor_grid.tiers[0].intervals[i].start_time, doctor_grid.tiers[0].intervals[i].end_time))
        speaker_texts.append(doctor_grid.tiers[0].intervals[i].text)

    # sort lists by speaker_intervals
    ordering = sorted(range(len(speaker_intervals)), key=lambda k: speaker_intervals[k])
    speaker_labels = [speaker_labels[i] for i in ordering]
    speaker_intervals = [speaker_intervals[i] for i in ordering]
    speaker_texts = [speaker_texts[i] for i in ordering]

    # do some preprocessing on the text
    # replace ... with space
    speaker_texts = [re.sub(r'\.\.\.', ' ', x) for x in speaker_texts]
    # remove stuff inside <>
    speaker_texts = [re.sub(r'<.*?>', '', x) for x in speaker_texts]
    # remove punctuation except for '
    speaker_texts = [re.sub(r'[^\w\s\']', '', x) for x in speaker_texts]
    # lowercase and remove leading/trailing whitespace
    speaker_texts = [x.lower().strip() for x in speaker_texts]
    # there are some empty strings, remove them, and corresponding speaker_labels and speaker_intervals
    empty_indices = [i for i, x in enumerate(speaker_texts) if x == '']
    speaker_labels = [x for i, x in enumerate(speaker_labels) if i not in empty_indices]
    speaker_intervals = [x for i, x in enumerate(speaker_intervals) if i not in empty_indices]
    speaker_texts = [x for i, x in enumerate(speaker_texts) if i not in empty_indices]
    assert len(speaker_labels) == len(speaker_intervals) == len(speaker_texts)

    speaker_labels = ['1' if x == 'doctor' else '2' for x in speaker_labels]

    # split the text into words and thus also produce the word-level speaker labels
    ref_labels = []
    ref_times = []
    ref_words = []

    for i in range(len(speaker_texts)):
        words = speaker_texts[i].split()
        for j in range(len(words)):
            ref_labels.append(speaker_labels[i])
            ref_times.append(speaker_intervals[i])
            ref_words.append(words[j])

    return ref_labels, ref_words, ref_times

def aws_labels(f, start_time, end_time):
    """Extract speaker labels and spoken words from AWS transcript file."""
    aws = json.loads(open(f).read())
    labels_aws = []
    words_aws = []
    times_aws = []

    sp0 = None
    
    for item in aws['results']['items']:
        if item['type'] == 'pronunciation':
            word_start_time = float(item['start_time'])
            word_end_time = float(item['end_time'])
            if word_start_time < start_time:
                continue
            if word_end_time > end_time:
                break
            if not sp0:
                sp0 = item['speaker_label']
            labels_aws.append(item['speaker_label'])
            words_aws.append(item['alternatives'][0]['content'].lower())
            times_aws.append((word_start_time, word_end_time))
    labels_aws = ['1' if x == sp0 else '2' for x in labels_aws]
    return labels_aws, words_aws, times_aws

def azure_labels(f, start_time, end_time):
    """Extract speaker labels and spoken words from Azure transcript file."""
    azure = json.loads(open(f).read())['transcription'][0]['recognizedPhrases']
    labels = []
    words = []
    times = []

    sp0 = None
    for item in azure:
        if 'speaker' in item:
            item2 = item['nBest'][0]
            for word_dict in item2['words']:
                word_start_time = float(word_dict['offsetInTicks']) / 10000000
                word_end_time = word_start_time + float(word_dict['durationInTicks']) / 10000000
                if word_start_time < start_time:
                    continue
                if word_end_time > end_time:
                    break
                if not sp0:
                    sp0 = item['speaker']
                # remove punctuation - but not '
                word = re.sub(r'[^\w\s\']', '', word_dict['word']).strip().lower()
                if not word:
                    continue

                labels.append(item['speaker'])
                words.append(word)
                times.append((word_start_time, word_end_time))
    labels = ['1' if x == sp0 else '2' for x in labels]
    return labels, words, times

def whisperx_labels(f, start_time, end_time):
    """Extract speaker labels and spoken words from WhisperX transcript file."""
    whisperx = json.loads(open(f).read())['word_segments']
    labels = []
    words = []
    times = []

    sp0 = None    
    for item in whisperx:
        if 'speaker' in item:
            word_start_time = float(item['start'])
            word_end_time = float(item['end'])
            if word_start_time < start_time:
                continue
            if word_end_time > end_time:
                break
            if not sp0:
                sp0 = item['speaker']
            labels.append(item['speaker'])
            # remove punctuation - but not '
            word = re.sub(r'[^\w\s\']', '', item['word']).strip().lower()
            words.append(word)
            times.append((word_start_time, word_end_time))
    labels = ['1' if x == sp0 else '2' for x in labels]
    return labels, words, times

def gcp_labels(f, start_time, end_time):
    """Extract speaker labels and spoken words from GCP transcript file."""
    gcp = json.loads(open(f).read())['alternatives'][0]['words']
    labels = []
    words = []
    times = []

    sp0 = None    
    for item in gcp:
        # start time format - start_time: '42.0s'
        word_start_time = float(item['start_time'].split('s')[0])
        word_end_time = float(item['end_time'].split('s')[0])
        if word_start_time < start_time:
            continue
        if word_end_time > end_time:
            break
        if not sp0:
            sp0 = item['speaker_tag']
        labels.append(item['speaker_tag'])
        words.append(item['word'].lower())
        times.append((word_start_time, word_end_time))
    labels = ['1' if x == sp0 else '2' for x in labels]
    return labels, words, times

## recreate helper functions - adjusted for OpenWillis-Diarize from https://github.com/google/speaker-id/blob/master/DiarizationLM/diarizationlm/utils.py#L189
def create_diarized_text(
    word_labels,
    ref_labels,
    use_new_line = False) -> str:
  """Create diarized text from words and speaker labels."""
  output = []
  previous_speaker = None
  for word, speaker in zip(word_labels, ref_labels):
    if speaker != previous_speaker:
      if previous_speaker and use_new_line:
        output.append("\n")
      output.append('<spk:' + speaker + '>')
    output.append(word)
    previous_speaker = speaker
  return " ".join(output)

def extract_text_and_spk(
    completions: str
) -> tuple[str, str]:
  """Extract the text and spk from the completions string."""
  spk = "1"
  previous_spk = "1"
  result_text = []
  result_spk = []
  for word in completions.split():
    if word.startswith("<spk:"):
      if not word.endswith('>'):
        word += '>'
      spk = word[len("<spk:"):-len('>')]
      # Handle undefined behaviors of non-recognizable spk with a placeholder.
      try:
        spk_int = int(spk)
        if not spk or spk_int < 1 or spk_int > 10:
          raise ValueError("Seeing unexpected word: ", word)
        previous_spk = spk
      except ValueError:
        print("Skipping meaningless speaker token:", word)
        spk = previous_spk
    else:
      result_text.append(word)
      result_spk.append(spk)
  return " ".join(result_text), " ".join(result_spk)
## end recreate helper functions

def format_data(sample):
    """Format the sample data into a prompt."""
    instruction = f"### Instruction\n{sample['instruction']}"
    response = f"### Answer\n{sample['response']}"

    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, response] if i is not None])
    return prompt
