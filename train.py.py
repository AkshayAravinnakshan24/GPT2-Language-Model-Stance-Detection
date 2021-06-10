# -*- coding: utf-8 -*-

"""

class Star_Stance_Model():
     





    ########################################################################
    #Convert to the required format of the GPT2 model
    class DataConverter():
      def __init__(self,dataset, train, use_tokenizer):

        
        self.texts = []
        self.labels = []
        
        if train ==True:
          self.texts = dataset.trX
          print(len(self.texts))
          self.labels= dataset.trY
        elif train== False:
          self.texts = dataset.vaX
          self.labels= dataset.vaY
        else:
          self.texts = dataset.teX
          self.labels= dataset.teY  
        
        self.n_examples = self.labels 
        return
        
      def __len__(self):
        
        return (len(self.n_examples))

      def __getitem__(self, item):


        #Returns: Dictionary of inputs that contain text and asociated labels.
        return {'text':self.texts[item],
                'label':self.labels[item]}


    #colate function for dataloader
    class Gpt2Collator:
          
          def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

            # Tokenizer to be used inside the class.
            self.use_tokenizer = use_tokenizer
            # Check max sequence length.
            self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
            # Label encoder used inside the class.
            self.labels_encoder = labels_encoder

            return
            
          def __call__(self, sequences):
            
            import torch
            # Get all texts from sequences list.
            texts = [sequence['text'] for sequence in sequences]
            # Get all labels from sequences list.
            labels = [sequence['label'] for sequence in sequences]
            # Encode all labels using label encoder.
            #labels = [self.labels_encoder[label] for label in labels]
            # Call tokenizer on all texts to convert into tensors of numbers with 
            # appropriate padding.
            inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
            # Update the inputs with the associated encoded labels as tensor.
            inputs.update({'labels':torch.tensor(labels)})
            
            return inputs

    #class to load my new dataset
    class Inputdatareader:
  
      def __init__(self, topic=None):
        seed = 3535999445
        #Provide the path to the Input data file
        
        self.teX,self.teY = self.stance( topic=topic)
    
      # Cleaning the tweets and converting labels to numbers.
      def stance(self, topic=None):
        import torch
        import pandas as pd
        import numpy as np
        path='/content/drive/MyDrive/BT_data/patagonia dataset - Copy.csv'
        def clean_ascii(text):
            # function to remove non-ASCII chars from data
            return ''.join(i for i in text if ord(i) < 128)
        orig = pd.DataFrame(columns=['text', 'Labels'])
        #orig = pd.read_csv(path, encoding = "utf-8")
        #Generate random values for True Labels. Because the model need something as input. But we just ignore it later.
        orig['text']=[Input_tweet]
        orig['labels']=np.random.randint(0,2, size=len(orig['text']))
        orig['text'] = orig["text"].apply(clean_ascii)
        X = orig['text']
        Y= orig['labels']   
        return X,Y

    # push the patagonia dataset through the model
    def test(self,dataloader,saved_model, device_):
      
      from tqdm.notebook import tqdm
      import torch
      import math
      # Use global variable for model.
      global model
      #the_model = torch.load('/content/drive/MyDrive/BT_data/model2.pt', map_location=torch.device('cpu'))
      # Load saved model ('the_model') instead of new model
      model= saved_model

      predictions_labels = []
      true_labels = []
      total_loss = 0
      all_probs=[]
      sum=0.0

      # Put the model in evaluation mode
      model.eval()
      
      # Evaluate data for one epoch
      for batch in tqdm(dataloader, total=len(dataloader)):

        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
        with torch.no_grad():   
            # Send batch of new data to the model      
            outputs = model(**batch)
            # Model outputs obtained are loss and Logit
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            
            # Converting logits(logistic regression) to Probabilities(similar to softmax)
            for i in logits:
              #logits1=max(i)
              sum=0.0
              for j in i:
                sum+= float((math.exp(j)))
              probs=[]
              prob=0.0
              for j in i:
                prob=math.exp(j)/sum
                probs.append(prob)
            print(type(probs))

            # update list
            predictions_labels += predict_content
          
      # Calculate the average loss over the training data.
      avg_epoch_loss = total_loss / len(dataloader)
      
      #De-Tokenizer and function to determine overall public opinion towards this topic
      labels=predictions_labels
      if predictions_labels==[0]:
        predictions_labels='AGAINST'
      elif predictions_labels==[1]:
        predictions_labels='FAVOR'
      elif predictions_labels==[2]:
        predictions_labels='NONE'
 
      # Return all true labels and prediciton for future evaluations.
      return predictions_labels,probs
    
    def imports_and_installs(self):
        
        #Install all the required packages
        !pip install transformers
        !pip install numpy as np
        !pip install pandas as pd
          


    #def return_values(self, a, b,c):
    # return a, b, c


    def __init__(self,Input_tweet1):
        
        
        
        #Input tweet
        #self.imports_and_installs()
        

        #Import all the Libraries
        import os
        import csv
        import io
        import math
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import Dataset, DataLoader
        from pathlib import Path
        from tqdm.notebook import tqdm
        from transformers import (set_seed,TrainingArguments,Trainer,GPT2Config,GPT2Tokenizer,AdamW,get_linear_schedule_with_warmup,GPT2ForSequenceClassification)
    
        global Input_tweet
        Input_tweet=Input_tweet1
        # Setting model parameters
        epochs = 4
        batch_size = 8
        max_length = 60
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name_or_path = 'gpt2'
        labels_ids = {'AGAINST': 0, 'FAVOR': 1, 'NONE' : 2}
        n_labels = len(labels_ids)

        # download model, config, tokenizer etc

        # Get model configuration.
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

        # Get model's tokenizer.
        
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token


        # Get the actual model.
        model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id

        # Load model to defined device.
        model.to(device)


        #Load the saved model
        #Sentiment Hugging Face
        the_model = torch.load('/content/drive/MyDrive/BT_data/model2.pt', map_location=torch.device('cpu'))
        ###############################################################################
    
        #Dataloader for patagonia dataset 
        gpt2_classificaiton_collator = self.Gpt2Collator(use_tokenizer=tokenizer, labels_encoder=labels_ids, max_sequence_len=max_length)
        train_dataset = self.Inputdatareader()
        gold_dataset =  self.DataConverter(train_dataset,train=None, use_tokenizer=tokenizer)
        gold_dataloader= DataLoader(gold_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
        #global self.Stance_,self.Probability_Scores_
        self.Stance_,self.Probability_Scores_ = self.test( gold_dataloader,the_model, device)
        #print('This tweet expresses a ' + str(Stance_)+' stance towards this topic and the Probability_Score= '+ str(Probability_Score_))
        #self.return_values(str(Stance_))
        
        return 
        
        ###############################################################################
