# GPT2-Stance-Analysis
## Usage
1. The full code used for model training is saved in train.py. 
2. In this repository, I have attempted to recreate the **'pipeline architecture'** for my GPT2 model. With this architecture you can run my huge model on the CPU itself and thereby efficiency of my model.    
## 1. GPT2
GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data. GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. I have added two more dense layers to this state of art model and converted it to perform Stance classification on twitter data. 

## 2. SETTING UP ENVIRONMENT 
### 2.1 USING ANACONDA ENVIRONMENT FILE

You can also Import the Anaconda environment directly into the local machine. The file needed to create an Anaconda environment is stored as ‘EnvironmentImportFile.yml’. The setup of the same can be done in the following ways:

1. Open anaconda prompt and deactivate the activate environment using 
```python
conda deactivate 
```
2. Navigate to the downloaded **‘EnvironmentImportFile.yml’** file in the prompt.

3. Run the command below with suitable environment name 

```python
conda env create -n <NAME> -f EnvironmentImportFile.yml 
```
4. Install and Open Jupyter Notebooks within this environment using 
```python
pip install notebook
jupyter notebook
```
5. You can also manually install all the pre-requisties listed below and then access Jupyter Notebooks in that environment

### 2.2 MANUAL SETUP OF ENVIRONMENT
```python
1. pip install transformers
2. pip install torch
3. pip install numpy as np
4. pip install pandas as pd
5. conda install -c conda-forge ipywidgets
6. jupyter nbextension enable --py widgetsnbextension
 
```

## 3. USING THE MODEL
I have fine-tuned the Hugging face English GPT2 model on Tesla T4 Azure VM. I have also open sourced the fine-tuned model. I order to see the working of this model please use the following steps.
1. Clone the repository with the following files.
2. Download my trained sentiment model parameters files to the same folder from my other repository 'https://github.com/AkshayAravinnakshan24/Sentiment-Detection-Model-Parameters-file.git'
3. Create new folder in Jupyter Notebooks in that environment and upload all the files into that folder. 
- ##### Note: Uploading the parameters file on Jupyter Notebooks takes a lot of time. Therefore, try to paste it in the current working directory. So that you can access it faster.
4. Replace the tweet variable with any desired tweet. (You can also provide urls, Hastags, and special charecters )
5. The model outputs are stored in parameters ‘Outputs.Stance_’ and ‘Ouputs.Probability_Score_’.

```python
!pip install import-ipynb
import import_ipynb,StarStancePredictor
from StarStancePredictor import Star_Stance_Model
tweet= ‘Crazy: ‘Patagonia's viral 'vote the #assholes out' tags are real! Read more on: https://t.co/N2fa0MxlOG #patagonia #viral #VoteThemAllOut2020 #VoteHimOut #VoteHimOut2020 #VoteReady #VoteThemOut #VoteTrumpOut #VoteTheAssholesOut #AssholesLiveForever #votetosaveamerica #Hayvine https://t.co/rEn5uEs8io’
Output =StarStancePredictor.Star_Stance_Model(tweet)
print(str(Output.Stance_)+', '+ str(Output.Probability_Scores_[1])+', '+ str(Output.Probability
_Scores_[0])+', '+ str(Output.Probability_Scores_[2]))

```
