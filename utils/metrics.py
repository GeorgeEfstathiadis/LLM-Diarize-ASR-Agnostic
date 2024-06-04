import meeteval.wer

def calculate_swer(text1, spk1, text2, spk2):
    """Calculate SA-WER between two speaker diarizations"""
    hyp1 = ""
    hyp2 = ""
    for i in range(len(text1.split())):
        speaker = spk1.split()[i]

        if speaker == '1':
            hyp1 += ' ' + text1.split()[i]
        else:
            hyp2 += ' ' + text1.split()[i]
    hypothesis = [hyp1, hyp2]

    ref1 = ""
    ref2 = ""
    for i in range(len(text2.split())):
        speaker = spk2.split()[i]

        if speaker == '1':
            ref1 += ' ' + text2.split()[i]
        else:
            ref2 += ' ' + text2.split()[i]
    reference = [ref1, ref2]
    
    wer1 = meeteval.wer.wer.siso.siso_word_error_rate(
        reference=reference[0],
        hypothesis=hypothesis[0]
    ).error_rate * 100
    wer2 = meeteval.wer.wer.siso.siso_word_error_rate(
        reference=reference[1],
        hypothesis=hypothesis[1]
    ).error_rate * 100
    
    return (wer1 + wer2) / 2

def calculate_cpwer(text1, spk1, text2, spk2):
    """Calculate cpWER between two speaker diarizations"""
    hyp1 = ""
    hyp2 = ""
    for i in range(len(text1.split())):
        speaker = spk1.split()[i]

        if speaker == '1':
            hyp1 += ' ' + text1.split()[i]
        else:
            hyp2 += ' ' + text1.split()[i]
    hypothesis = [hyp1, hyp2]

    ref1 = ""
    ref2 = ""
    for i in range(len(text2.split())):
        speaker = spk2.split()[i]

        if speaker == '1':
            ref1 += ' ' + text2.split()[i]
        else:
            ref2 += ' ' + text2.split()[i]
    reference = [ref1, ref2]
    
    cpwer = meeteval.wer.wer.cp.cp_word_error_rate(
        reference=reference,
        hypothesis=hypothesis
    ).error_rate * 100
    
    return cpwer

def preprocess_str(text):
    """Preprocess a string for WER calculation"""
    text = text.replace('\n', ' ').replace('\\', '')
    # remove punctuation - except for "'"
    text = ''.join([c for c in text if c.isalnum() or c in [' ', "'"]])
    # remove multiple spaces
    text = ' '.join(text.split())
    text = text.lower().strip()
    return text
