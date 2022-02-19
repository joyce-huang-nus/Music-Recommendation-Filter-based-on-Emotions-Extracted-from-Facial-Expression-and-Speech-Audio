# Music-Recommendation-Filter-based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio

### 1. Introduction
“Music is a universal language, but we don’t always pay enough attention to what it’s saying and how it’s being understood.”                                                                                                                                    ——Alan Cowen 
Music is universally recognized as an effective way for humans to express emotion and regulate emotional states. Studies show that people can evoke 13 distinct emotions from listening to music. Therefore, besides daily refreshment, music is also widely used in professional areas such as music therapy and music education. The question is, how to choose suitable music? 
Most of the existing music recommendation engines are based on collaborative preference or music content, without considering human emotions. However, considering the old or the sick who need more delicate emotional care, and people who are eager to have more interactions when quarantined during the Covid-19 pandemic, we think that what people need is not only music but the music fitted for the current emotion.
Therefore, we aim to create an emotion-based music recommendation model layer to better empower the existing system, which could bring lots of value. From the users’ perspective, it provides a sense of compassion and navigates listeners to a more positive emotional state, improving the user experience and gaining user satisfaction. From the business side, high satisfaction means higher customer stickiness, as well as easier acquisition. This emotion-based layer could be embedded in various products, no matter software or hardware. For example, we already see many AI assistants, like Amazon Alexa, Google assistant, Xiaodu, Xiaoai, etc. One can interact with it in apps, also smart speakers, and home automation. During this implementation process, with the model trained and the data gathered, the business side could learn more about the user's behavior and psychology. Finally, the techniques, the products (both software and hardware), and various use cases together help the business to form its smart industrial ecology. Different from the previous ecology, this integration is more emotion-based, generating a more user-friendly environment.

### 2. Problem Statement 
In this proposal, we would like to apply image and audio analytics techniques to detect users’ emotions. Combined with the existing music categories, we will propose an emotion-based music recommendation framework that can be applied to AI smart companion robots.

<img width="472" alt="image" src="https://user-images.githubusercontent.com/88580416/154793616-e7849f99-8df7-418c-9d59-15d758ad742d.png">
Figure 1: Framework of the Emotion-based Music Recommendation Layer

The main objectives of this project include:
>> Build a deep learning model to identify the emotion from the user's facial expression.
>> Build a deep learning model to detect the emotion from the user’s speech audio.
>> Leverage the facial and audio classification result to infer the real-time emotion of the user, and map the user’s emotion to the pre-labeled music database.
>> Frame the outline of the interactive multiple feedback of the music recommendation system.

### 3. Facial Expression Recognition

#### 3.1 Data Preparation
Our facial expression dataset is Facial Expression Recognition Challenge Dataset from Kaggle. This dataset has a uniform size for all images, 48x48 pixels. Pixel values of an image are extracted and stored in a csv file. Due to resources and time constraints,  we will use the pre-processed csv file to conduct descriptive analysis. 

#### 3.2 Dataset Description
The data consists of 48x48 pixel grayscale images of faces. There are 7 emotion categories, including anger, disgust, fear, happiness, sadness, surprise and neutral, in this dataset. There are 35,887 images in the csv file, and there are 3 columns, emotion, usage and pixels.

<img width="293" alt="image" src="https://user-images.githubusercontent.com/88580416/154793658-8ec4ad8a-d626-439e-8327-70d25744f536.png">
Figure 2: Facial Dataset Description

Emotion: 0 = anger, 1 = disgust, 2 = fear, 3 = happiness, 4 = sadness, 5 = surprise, 6 = neutral
Usage: 28,709 images used as Training, 3,589 images used as PrivateTest and 3,589 images used as PublicTest 
Pixels: each image has 2,304 (48x48) pixels

#### 3.3 Exploratory Data Analysis (EDA)

<img width="251" alt="image" src="https://user-images.githubusercontent.com/88580416/154793690-1ce95137-7c5c-4e7e-9a3e-6aff7dd8e681.png">
Figure 3: Number of Images by Emotion

Each emotion is imbalanced in this dataset. There is not enough ‘disgust’ emotion data, so we decide only to use the other 6 emotions, including anger, disgust, fear, happiness, sadness, surprise, and neutral, in our analysis.
We also check the proportions of each emotion in training, validation and test data and find they are similar. Please refer to Appendix I. 

#### 3.4 Modeling - Convolutional Neural Networks 
This is a classification problem and the most promising machine learning tool for image-recognition is Convolutional Neural Networks (CNNs). There are high-dimensional inputs, and the need for non-linear feature transformations. A CNN is a special case of the neural network, and it consists of one or more convolutional layers, and one or more fully connected layers in a standard neural network. We will follow typical CNNs models to try 5 to 25 distinct layers of pattern recognition and find the best output predictions. Figure 4 is the sketch of image recognition problems using a CNN, consisting of image inputs, convolutional layers, fully connected layers and output predictions.  

<img width="419" alt="image" src="https://user-images.githubusercontent.com/88580416/154793700-bcf28d3b-c7ac-4f5c-9dd3-9df21db0be53.png">
Figure 4. Sketch of Convolutional Neural Networks (CNNs)

### 4. Audio Emotion Recognition
#### 4.1 Audio Emotion Dataset

<img width="266" alt="image" src="https://user-images.githubusercontent.com/88580416/154793727-fc56eea4-c1f8-4fdb-b922-f839a6121403.png">

For the audio-emotion dataset, we use the RAVDESS Emotional speech audio dataset from Kaggle. This dataset contains 1440 speech audio files collected from 24 actors (12 males and 12 females). Each actor will record 60 audio trials with two lexically-matched statements in a neutral North American accent. The audio files have already been labeled with some features as below:

1. Modality: 01 = full-AV, 02 = video-only, 03 = audio-only. All files in this dataset are audio-only files.
2. Vocal channel: 01 = speech, 02 = song. All files in this dataset are speeches.
3. Emotion: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised. We will take the universal six emotions, including anger, disgust, fear, happiness, sadness, surprise and neutral.
4. Emotional intensity: 01 = normal, 02 = strong. Note that there is no strong intensity for the 'neutral' emotion.
5. Statement: 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door".
6. Repetition: 01 = 1st repetition, 02 = 2nd repetition.
7. Actor: 01 to 24. 
8. Gender: male and female.

There are 192 files in each of the emotions in the dataset, except the neutral emotion containing only 96 files. To analyze and vectorize the audio dataset, we intend to use the Python package Librosa. Librosa can load the audio file as time-series data. Therefore, we expect to get a dataframe with several audio metrics, with each metric containing a series of panel data, forming an audio feature matrix.

#### 4.2 Audio-Features Extraction
Audio features such as pitch, intensity, spectral energy distribution, average zero crossing density (ZCD), jitter, and MFCC have been discovered to be useful in emotion recognition. However, using only one audio feature is inefficient to train emotion-recognition models. The trend of audio-features extraction is to combine several complementary audio features. Despite feature selection and engineering, model accuracy rate is also found to be correlated to the number of emotions to be classified. Many models with higher accuracy rates classify only 2-3 emotion classes, indicating a tradeoff between model granularity and accuracy rate. However, to build a more practical and robust emotion recognition model, we intend to classify the emotions into the six universal emotions. We intend to extract the audio features combining 2 paths, the prosodic features and the spectral audio features, and we will then combine the two extraction paths with weighted average.

Table 1. Audio Feature Metrics Description

<img width="609" alt="image" src="https://user-images.githubusercontent.com/88580416/154793750-47172a6a-3bfc-444d-bd57-52990489b127.png">

Mean values for pitch features are shown in Figure 5 and averaged pitch signals for each emotion on the Hamming window are shown in Figure 6. The below graphs displays a great discrimination between the six emotions.

<img width="423" alt="image" src="https://user-images.githubusercontent.com/88580416/154793757-4aba5b11-525c-4129-92af-d6f1d84715c0.png">
Figure 5. Mean Values for Pitch                   Figure 6. Hamming Window

<img width="608" alt="image" src="https://user-images.githubusercontent.com/88580416/154793829-f89e8171-a5bb-4a56-b679-559c5a3443a4.png">

#### 4.3 Modeling
During the searching of possible model training methods for audio datasets and emotion recognition, we found some candidate methods for audio emotion recognition as following:
1. Support Vector Machine (SVM)
2. Regression modeling
3. Long-short term memory models (LSTMs)
4. Convolutional Neural Networks (CNNs)
5. Recurrent Neural Networks (RNNs)

The above methods are all possible to bring us ideal results, and we have not discovered a model that has extremely better performance than others in audio emotion recognition. Thus, although there seems to be more and more applications in Neural Network applications on audio datasets in recent years, we prefer to train each of the listed methods, evaluate the output performances, and make the final model selection.

### 5. What Music to Recommend?
#### 5.1 Music Emotion Recognition Dataset
In order to build the framework of the music recommendation system, music datasets with emotional labels are introduced. There is a broad range of music datasets labeled with emotional tags available. However, the annotation strategies are various among different datasets which requires data wrangling before using.  
Emotify contains 400 songs over four genres (Rock, Classical, PoP and Electronic). The annotations were collected using GEMS scale (Geneva Emotional Music Scales). Each participant could select at most three labels from the scale felt strongly when listening. Emotify was launched on the 1st of March 2013, and up to now 1778 people participated. The description of the emotional categories is summarized in Table 2 in appendix. Since some pieces of music show a combination of emotions, multi-labeled annotation gives a more comprehensive portrait of the music. 

The dataset collects original responses by each participant to one or more tracks. 7888 records are presented.
1. ID of the music file
2. Genre of the music file, i.e. Rock, Classical, PoP, Electronic.
3. The 9 emotion annotations chose by the participant
4. Participant's mood intensity prior to listening to the music.
5. Liking (1 if the participant reports he/she liked the song).
6. Disliking (1 if the participant reports he/she disliked the song).
7. Demographic features, including age, gender, and mother tongue. (self-reported)

For future data exploratory, we are expecting to discover some interesting aspects within the dataset as below. 
1. Is there a significant correlation between music genres and emotions? 
2. Is there any association between demographic features and emotions?
3. What kind of emotions tends to co-exist when aroused by music? Why?
Exploring the questions above could provide a deeper understanding of the dataset and even gives us insights on further research and application. 

#### 5.2 Output Emotion Integration
We proposed three possible ways to integrate emotion results generated from facial and audio data.
Train the two models separately, and combine the output similar to an ensemble model.
Have the two networks (if both datasets are trained using Neural Network) separated until making a combination layer somewhere before the output layer.
Build a new neural network using the logic and algorithms of the existing two neural networks.

#### 5.3 Emotion matching
<img width="216" alt="image" src="https://user-images.githubusercontent.com/88580416/154793837-ed33391b-2338-4154-84e0-6ddea1794864.png">
Figure 7: Valence-Arousal of Emotions
The ultimate goal of emotion-based music recommendation is to guide the user to a more positive emotional state. However, Mou’s research suggests that recommending a piece of music that has a similar Valence-Arousal value to the user’s self-reported emotional state as the first piece of music that the user listens to. This expression of similar emotion can give the user a feeling of compassion. Especially when a user is experiencing a depressed mood, the attempt to positively influence the user by a piece of music showing compassion may be more effective than directly playing an exciting or joyful one. In the common approach, we can directly match emotions or calculate the similarity of emotions in the Arousal-Valence model.

#### 5.4 Music Recommender Enabling
Through facial expression recognition and audio emotion recognition systems, information about the user's current emotion is fed to any conventional collaborative or content-based music recommender as supplementary data. We assume two ways for music recommendation engines to collaborate with user emotion data.
Algorithm boost: Feature matrix integration
Real-time emotion data presented in categorical forms are integrated into other model features. When the user gives feedback (thumb, score…), labels are generated. Recommender then retrains the model to learn the user preference pattern with moods. The recommender performs better as users interact more with our AI assistants. 
Emotion time: Directly mapping music with current emotion status
The second way is directly mapping music with the user’s current mood through a matching algorithm. As mentioned above, emotions can be deconstructed into 2 dimensions (Arousal and Valence). All emotional statuses will be coordinated in 2D and mapped with labeled music in the dataset. The mapping process required music therapist domain knowledge. To simplify this complex process, our recommender maps music in a similar mood to resonate with users. 

### 6. Discussions
#### 6.1 More Applications
Despite the core application to embed this layer into various software and hardware’s music recommendation systems, and to generate better user experience as well as business values, our proposal could also be used in music therapy and psychological counseling, complementing the traditional subjective judgments. If we further jump out of the music recommendation, the ML solution of recognizing emotion from facial/audio data could be useful in many other domains, like improving the quality of telecom services, detecting lies of suspects, assisting in safe driving, etc.

#### 6.2 Future Works
1. There is a trade-off: A finer granular of emotions decrease the model accuracy, while a general category limits business value. This should be calibrated according to different use cases.
2. More psychological domain knowledge is needed to help with matching the user's emotion with music. 
3. Finally, there are more interpretations to be done. For example, if we could learn the mechanism of music labeling, i.e. what characteristic of music contributes to lit up a user's emotion, then it can help compose the particular music for music therapy.

### References
[1] L. Mou, J. Li, J. Li, F. Gao, R. Jain and B. Yin, "MemoMusic: A Personalized Music Recommendation Framework Based on Emotion and Memory," 2021 IEEE 4th International Conference on Multimedia Information Processing and Retrieval (MIPR), 2021, pp. 341-347, doi: 10.1109/MIPR51284.2021.00064.
[2] F. Kuo, M. Chiang, M. Shan and S. Lee, "Emotion-based music recommendation by association discovery from film music", Proceedings of the 13th annual ACM international conference on Multimedia. Association for Computing Machinery, pp. 507-510, 2005.
[3] D. Ayata, Y. Yaslan and M. E. Kamasak, "Emotion Based Music Recommendation System Using Wearable Physiological Sensors," in IEEE Transactions on Consumer Electronics, vol. 64, no. 2, pp. 196-203, May 2018, doi: 10.1109/TCE.2018.2844736.
[4] Song, Yading and Simon Dixon. “PREDICT THE EMOTIONAL RESPONSES OF PARTICIPANTS ?” (2015).
[5] F. H. Rachman, R. Samo and C. Fatichah, "Song Emotion Detection Based on Arousal-Valence from Audio and Lyrics Using Rule Based Method," 2019 3rd International Conference on Informatics and Computational Sciences (ICICoS), 2019, pp. 1-5, doi: 10.1109/ICICoS48119.2019.8982519.
[6] J. Kim, S. Lee, S. Kim and W. Y. Yoo, "Music mood classification model based on arousal-valence values," 13th International Conference on Advanced Communication Technology (ICACT2011), 2011, pp. 292-295.
[7] J. Bai et al., "Dimensional music emotion recognition by valence-arousal regression," 2016 IEEE 15th International Conference on Cognitive Informatics & Cognitive Computing (ICCI*CC), 2016, pp. 42-49, doi: 10.1109/ICCI-CC.2016.7862063.
[8] A. Kolli, A. Fasih, F. A. Machot and K. Kyamakya, "Non-intrusive car driver's emotion recognition using thermal camera," Proceedings of the Joint INDS'11 & ISTET'11, 2011, pp. 1-5, doi: 10.1109/INDS.2011.6024802.
[9] Morrison D, Wang R, De Silva L C. Ensemble Methods for Spoken Emotion Recognition in Call - Centers [ J ]. Speech Communication, 2007, 49(2):98-112.

