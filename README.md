# Collab Link for training Marathi model:
https://colab.research.google.com/drive/10T9VFTJ5uCg679Dg7IBQCIr7EQtMn_ZN?usp=sharing

## Dataset used from:
https://www.openslr.org/103/
VCTK Format

# Training_VITS_Hindi_TTS

## Introduction
This repository contains code for training a Text-to-Speech (TTS) model specifically for Hindi language using the VITS model. The VITS model is known for its high-quality speech synthesis capabilities. 

## Installation
```To get started with training the Hindi TTS model, follow these steps:
git clone https://github.com/Ishank56/vits_using_coqui.tts.git
pip install -e .
```
2. Ensure that the Hindi dataset is available inside the `Dataset` folder. The Hindi data i used can be downloaded from [here](https://www.iitm.ac.in/donlab/indictts/database). The data should be formatted in a manner similar to LJSpeech_1.1 dataset for compatibility.

Dataset I used for training here:
https://keithito.com/LJ-Speech-Dataset/

3. Install all required libraries for phonemizing Hindi alphabets. **Espeak** library is particularly useful for this purpose.
   For specific files config.json needs to be set accordingly while using vits model, 
5. Adjust the parameters in the code according to your requirements. The rest of the parameters should already be updated accordingly for the Hindi dataset.

## Dataset
The dataset provided in the `Dataset` folder contains text data and corresponding Wav files in Hindi language. It follows the same format as the LJSpeech 1.1 dataset for consistency. Ensure that the dataset is properly formatted and organized before proceeding with training.

## Training
To train the TTS model for Hindi language, run the `train_tts.py` file. This file contains the necessary code for training the model using the VITS architecture. Make sure all dependencies are installed and the dataset is properly configured before initiating the training process.

```python
nvidia-smi   #to check the GPUS available
CUDA_VISIBLE_DEVICES="5" python train.py  #Mention the GPU to run on and the training file
```

## Inference

```python
pip install TTS

tts --text "यह अपनत्व और उत्कर्ष गुलज़ार की पूरी ज़िदगी और उनके अनेक अन्य कार्यों में आसानी से लक्षित हो जा सकती है." \
    --model_path path/to/model.pth \
    --config_path path/to/config.json \
    --out_path folder/to/save/output.wav
```
or 
```python
python testing.py

```
## Contributing
I made the project working and it is generating good synthesised output. Contributions to this project are welcome! If you have any improvements or suggestions, feel free to open an issue or submit a pull request.

## License
I give credits to https://github.com/coqui-ai/TTS?tab=readme-ov-file and dataset contributors mentioned above.
