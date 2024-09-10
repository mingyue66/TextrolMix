# TextrolMix
Welcome to the official webpage for the TextrolMix dataset. This dataset aims to support text-guided Target Speaker Extraction (TSE). 


Listen to the demo produced by the model trained on this dataset!  [![ICASSP Paper Demo]( https://img.shields.io/badge/StyleTSE-Demo-blue)](index.html)


<!-- <br>
<img src="textrolspeech_fig1.png">
<br> -->

## About 
The TextrolMix dataset is derived from the public dataset [Textrolspeech](https://github.com/jishengpeng/TextrolSpeech), which contains pairs of speech and text description, originally designed for text-prompt text-to-speech (TTS). We generate speech mixtures and provide two types of clues for target speaker extraction (TSE). This adaptation supports a flexible TSE application that focuses on speaking style over speaker identity.

**Clue Type I** [text-only] "Extract the male speaker with his shocked usual pitch adopts a slow-speaking rate."

**Clue Type II** [audio-text] A reference audio with surprised emotion, and the text "Isolate the speech with the same emotion as the reference."

## Statistics
- **Number of Samples**: 121486 two-talker mixtures, 157 hours, mean duaration: 4.65 seconds.

- **Mixture quality**: SNR ~ Normal(0,4) dB, Mean STOI: 0.71, Mean PESQ: 1.60

- **Style attribute**: speaker identity, gender, pitch, emotion, accent 

- **Clue Type I text description**: Original style descriptions are augmented with template-based mid and short versions. Training samples feature multiple descriptions per length, while validation and test samples have one fixed description each.

```csv
| Length  | Avg words |               Example                     |
|---------|-----------|-------------------------------------------|
|  Long   | 13        | The female speaker's customary tone       |
|         |           | complements her happy slow-paced speech.  |
|  Mid    | 6         | A lady’s talking happily.                 |   
|  Short  | 4         | Woman's voice.                            |
```

- **Clue Type II style attribute**: 
```csv
|     Attribute    | Percentage |          Attribute values          |
|------------------|------------|------------------------------------|
| Speaker identity | 52%        | [individual speakers over 1300]    |
|  Emotion         | 16%        | sad, surprised, angry, happy, fear,|
|                  |            | contempt, disgusted, neutral       |
|  Gender          | 11%        | female, male                       |   
|  Accent          | 11%        | american, british, scottish, irish |
|                  |            | south african, canadian, australian|
|  Pitch           | 10%        | high, normal, low                  |

Note: accent is from source corpora.
```

- **TSE performance on TextrolMix test set**: SI-SDRi (dB)
```csv
| Model  |Long|Mid |Short|Emotion|Accent|Pitch|Gender|SpeakerID|
|--------|----|----|-----|-------|------|-----|------|---------|
|StyleTSE|16.5|16.3| 16.4| 15.9  |19.6  |18.8 |18.2  |15.7     |
```

## Getting Started 

**1. Download the Textrolspeech dataset:**
Follow the open-source download link for [Textrolspeech](https://github.com/jishengpeng/TextrolSpeech) to download the speech data. Text data is not needed to generate TextrolMix.

**2. Download the mixture metadata:**
Download the TextrolMix metadata csv files (```train_meta.csv, dev_meta.csv, test_meta.csv```). Metadata contains paths to component utterances and clues for identifying and extracting target speech.

Each mixture sample in the metadata contains the following information:

```csv
| mixed_fp | target_fp | background_fps | style_prompt_long | style_prompt_mid | style_prompt_short | audio_clue_fp | clue_attribute | attribute_value | text_clue | target_loudness | background_loudness | duration_mixture_min | duration_mixture_max |
```

**3. Organize data:**
Place the downloaded speech data and TextrolMix metadata into the specified directory.


```
├── generate_textrolmix_from_metadata.py
├── metadata/
  ├── train_meta.csv
  ├── dev_meta.csv
  └── test_meta.csv
├── textrolspeech/ 
  ├── libritts/
  ├── VCTK-Corpus/
  └── emotion_dataset/
    ├── MEAD/
    ├── ESD/
    ├── MESScompressed/
```


**4. Generate mixture wavefiles:**
Run the script ```generate_textrolmix_from_metadata.py``` to automatically resample, normalize loudness, and mix the audio. Adjust the sampling frequency and mixing mode within the script as needed.


The generated mixtures and resampled individual utterances will be saved to the "textrolmix" folder.

```
├── textrolmix/ 
  ├── single/
  └── mix/
    ├── train/
    ├── dev/
    └── test/
```


<!-- <br>
<img src="figure3.png">
<br> -->



## Citations
If you use TextrolMix, please cite this webpage in the footnote.

<!-- If you only use Textrolspeech, please cite the following paper:

```bibtex
@inproceedings{ji2024textrolspeech,
  title={Textrolspeech: A text style control speech corpus with codec language text-to-speech models},
  author={Ji, Shengpeng and Zuo, Jialong and Fang, Minghui and Jiang, Ziyue and Chen, Feiyang and Duan, Xinyu and Huai, Baoxing and Zhao, Zhou},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10301--10305},
  year={2024},
  organization={IEEE}
}
``` -->





