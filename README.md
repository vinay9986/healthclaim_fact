## Model
Bert model trained on PubMed articles by microsoft. Pretrained model weights can be downloaded from [here](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).

## Dataset
[PUBHEALTH](https://huggingface.co/datasets/health_fact#dataset-card-for-pubhealth): Explainable automated fact-checking of public health claims.

## Analysis:
### Stage 1: Establishing the baseline.
* Fine tune the bert model to the *claim* feature of the pubhealth dataset. The baseline accuracy is **0.6502** with f-score **0.6072**
* Code in notebook **healthcare_claims_baseline.ipynb**

### Stage 2
* Pairwise learning of *claim* and *explanation* analysis
  * *explanation* feature is “abstractive explanation” generated from *main_text* feature as explained in the [paper](https://arxiv.org/abs/2010.09926)
  * This pairwise learning boosts the accuracy to **0.7263** with f-score **0.7088**
  * Code in notebook **healthcare_claims_with_explanation.ipynb**
* Pairwise learning of *claim* and *main_text* analysis
  * *main_text* feature is the complete fact notes written by expertes in the domain and is collected from the various fact checking websites
  * There is a slight bump up of accuracy compared to baseline. The accuracy is **0.6842** with f-score **0.6555**
  * The reason why this pair does not provide significant improvement of accuracy might be because the noise present in the *main_text* feature
  * Code in notebook **healthcare_claims_with_main_text.ipynb**

### Stage 3
Pairwise learning of *claim* and sentences similar to claim from *main_text* analysis
* First preprocess *claim_id* and *main_text*
* break the *main_text* into sentences using the notebook **healthcare-preprocess-and-generate-data.ipynb**
* Next extract the embeddings of CLS token from the last layer of Bert model for *claim* and sentences of *main_text* using notebook **healthcare-generate-embeddingd.ipynb**
* Next generate the train, validation and test datasets that contains *claim*, top 5 similar sentences to claim from *main_text* and label using notebook **healthcare-cosine-similarity-analysis.ipynb**
* Bert model fine tuned on generated dataset has accuracy of **0.6729** with f-score **0.6355**
* Code in notebook **health-claim-with-main-text-top5-similar-sentences.ipynb**
* Looks like the generated abstractive explanation of *main_text* is the best representation of *main_text* feature

### Stage 4
Pairwise learning of *claim* and *explanation* features with custom head over BertModel
* The performance of the model is similar to default model
* Code in notebook **healthcare-claims-with-explanation-custom-head.ipynb**

## Further work
* Try different representations of the hidden layer embeddings with *claim* and *explanation* pair
* Try machine learning models over embeddings explored in the previous step
