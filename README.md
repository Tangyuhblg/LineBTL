# LineBTL: A Multi-Granularity and Context-Aware Approach for Line-Level Defect Prediction
![LineBTL Overview](Fig. 3 LineBTL overview.pdf)

## Datasets 
The datasets are obtained from Wattanakriengkrai et. al. The datasets contain 32 software releases across 9 software projects. The datasets that we used in our experiment can be found in this [github](https://github.com/awsm-research/line-level-defect-prediction).
## Repository Structure
Our repository contains the following directory
- output: This directory contains the following sub-directories

  * Word2Vec_model: This directory stores word2vec models of each software project

  * model: This directory stores trained models

  * prediction: This directory stores prediction (in CSV files) obtained from the trained models
        
- src: This directory contains the following directories and files

  * code_preprocessing.py: The source code used to preprocess datasets for file-level model training and evaluation
    
  * split_basic_block.py: Dividing into basic blocks
    
  * my_util.py: The source code used to store utility functions
 
  * train_word2vec.py: The source code used to train word2vec models
 
  * LineBB_model.py: The source code that stores LineBB architecture
 
  * train_model.py: The source code used to train LineBB models
 
  * test.py: The source code used to generate prediction (for RQ1 and RQ2)
 
  * generate_prediction_cross_projects.py: The source code used to generate prediction (for RQ4)
 
  * evaluation_index.py: Output evaluation indexs
 
  * llm_file_preprocessing.py: Source code preprocessing for LLM
 
  * ours_code_preprocessing.py: Write code to Java files
 
  * ours_run.py: Use LLM to output high-risk line numbers and write the results to the JSON file
 
  * ours_analysic_json.py: Analyze the json results of the LLM output
 
  * ours_evaluation_index.py: Output evaluation indexs (for RQ3)
- results: LLM output results.

# Experiment
## Experimental Setup
We use the following hyper-parameters to train our DeepLineDP model
- batch_size = 64

- num_epochs = 10

- embed_dim (word embedding size) = 150/200
  
- word_att_dim = 64

- word_gcn_hidden_dim = 128
  
- basic_block_att_dim = 64

- basic_block_gcn_hidden_dim

- line_att_dim = 64

- line_hidden_dim = 128

- dropout = 0.2

- lr (learning rate) = 0.001


