# Run this to download and save the sentence transformer library used for the website
get_ipython().system('pip install -U sentence-transformers')

from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

savePath = "./website/st_model"
sentence_transformer_model.save(savePath)