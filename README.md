# Speech-Emotion-Recognition
Speech is the most natural way of expressing ourselves as humans. It is only natural then to extend this communication medium to computer applications. We define speech emotion recognition (SER) systems as a collection of methodologies that process and classify speech signals to detect the embedded emotions. SER is not a new field, it has been around for over two decades, and has regained attention thanks to the recent advancements. These novel studies make use of the advances in all fields of computing and technology, making it necessary to have an update on the current methodologies and techniques that make SER possible. We have identified and discussed distinct areas of SER, provided a detailed survey of current literature of each, and also listed the current challenges.
# Dataset
Our [dataset](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en?select=Ravdess) is more than 7K audio files and each of them is 2-4 seconds. The whole dataset is more than 3GB.  
There are 6 different categroy: 
- SAD - Sadness
- ANG - Angry
- DIS - Disgust
- FEA - Fear
- HAP - Happy
- NEU - Neutral

# Data Cleaning
Since each file have different length, we need to adjust their length, we also need to chech the null value
<img src="audio.png" width="400">

```python
# check the length of audio and chop them into same length
def adjust_length(time_series_list, length):
    n = len(time_series_list)
    for i in range(n):
        audio_length = len(time_series_list[i])
        if audio_length < length:
            time_series_list[i] = np.append(time_series_list[i], [0 for i in range(length-audio_length)])
        else:
            time_series_list[i] = np.array(time_series_list[i][:length])
```

# Data Preprocessing and Feature Extraction (Audio -Domain Knowledge)

```python
def feature_extraction_1D(data):

    # Zero Crossing rate
    features = librosa.feature.zero_crossing_rate(y=data)

    # Energy
    features = np.append(features, librosa.feature.rms(y=data), axis=1)

    # Mel-frequency cepstral coefficient
    l = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=13), axis=0).reshape(1, 106)
    features = np.append(features, l, axis=1)
    
    # Spectral Centroid
    features = np.append(features, librosa.feature.spectral_centroid(y=data, sr=sampling_rate), axis=1)
    
    # Spectral Bandwidth
    features = np.append(features, librosa.feature.spectral_bandwidth(y=data, sr=sampling_rate), axis=1)
    
    # Spectral Flatness
    features = np.append(features, librosa.feature.spectral_flatness(y=data), axis=1)
    
    # Spectral Rolloff maximum frequencies
    features = np.append(features, librosa.feature.spectral_rolloff(y=data, sr=sampling_rate), axis=1)
    
    # Spectral Rolloff minimum frequencies
    features = np.append(features, librosa.feature.spectral_rolloff(y=data, sr=sampling_rate, roll_percent=0.01), axis=1)
    
    return np.array(features)
```


# Model and Result
We used Pytorch to build our model. We follow the structure of the ResNet but instead of using `nn.Conv2D` we change to `nn.Conv1D`.
For the Sequence Model, we also tried deploying Bidirectional GRU, Bidirectional LSTM and Attention model. The training speed was not as fast as CNN model, so we only trained 25 epochs. 

| Model | Accuracy |
|--------|--------|
| CNN(ResNet) | 0.92 |
| CNN(AlexNet) | 0.88 |
|Attention | 0.63|
|LSTM|0.53|
| GRU|0.41|



