Once clips are extracted, run automated models on each modality and store predictions alongside the human annotations. These become the machine-vs-human comparison layer of the benchmark. 

⚠  Automated predictions should be stored as separate columns in the master spreadsheet — never overwrite human annotation columns. The human gold labels are ground truth; model outputs are predictions to be evaluated against them. 

3.1 Facial expression — DeepFace 

Run DeepFace on the extracted frames for each TCU clip. Take the majority-vote prediction across all frames in the clip as the clip-level label. 

 

3.2 Vocal tone — SpeechBrain or a pre-trained emotion classifier 

Run a pre-trained speech emotion recognition model on each .wav clip. Recommended options: 

    SpeechBrain emotion recognition (speechbrain/emotion-recognition-wav2vec2-IEMOCAP) 

    openSMILE feature extraction + a simple classifier if SpeechBrain is too resource-intensive 

 

3.3 Stance — BERT-based text classifier 

Run a pre-trained sentiment or stance classifier on the tcu_transcript text. Recommended options: 

    cardiffnlp/twitter-roberta-base-sentiment-latest via HuggingFace Transformers 

    Or any fine-tuned stance detection model if available 

Map positive/negative/neutral labels to your codebook stance categories where possible. Store as auto_stance in the spreadsheet. Note this mapping is approximate — the model was not trained on political speech — but the comparison is still informative as a baseline. 

3.4 Output columns to add to master_clips.csv 

Column 
	

TCUID|auto_facial_mapped|auto_facial_frame_count|auto_tone_label|auto_tone_mapped|auto_tone_confidence|auto_stance_label|auto_stance_mapped|auto_stance_confidence 