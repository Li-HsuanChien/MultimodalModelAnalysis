

"""
Loop through each TCU:
    facial expression(Deep Face): Run through each frame, use model to get facial expression label, make majority vote
    Tone(SpeechBrain): Run through each wav, use model to get emotion label
    Text(BERT): Run through each tcu_transcript, use model to get sentiment/stance label

"""