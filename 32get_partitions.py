# 3.2 Program to create partitions of the cleaned unlabelled dataset
#divided into partitions to avoid exceeding the memory limit while creating embeddings

#install the  datasets,numpy and pandas packages using preferred installer program (pip)
pip install datasets #https://pypi.org/project/datasets/
pip install numpy # https://numpy.org/
pip install pandas # https://pandas.pydata.org/

#import the required libraries
from datasets import Dataset
import math #https://docs.python.org/3/library/math.html

#reading the cleaned unlabelled dataset csv file
df = Dataset.from_csv("df_clean_wo_embeddings.csv")

#creating partitions of the dataset for each team member to create embeddings for the review text
len = df.num_rows
len_partition = math.floor(len/6)
partition_1 = np.arange(len_partition)
partition_2 = np.arange(len_partition, 2*len_partition, 1)
partition_3 = np.arange(2*len_partition, 3*len_partition, 1)
partition_4 = np.arange(3*len_partition, 4*len_partition, 1)
partition_5 = np.arange(4*len_partition, 5*len_partition, 1)
partition_6 = np.arange(5*len_partition, len, 1)

partitions = [partition_1, partition_2, partition_3, partition_4, partition_5, partition_6]
for partition in partitions:
  print(partition)

df_1 = df.select(partition_1)
df_2 = df.select(partition_2)
df_3 = df.select(partition_3)
df_4 = df.select(partition_4)
df_5 = df.select(partition_5)
df_6 = df.select(partition_6)

df_1.to_csv('df_part_sit.csv')
df_2.to_csv('df_part_par.csv')
df_3.to_csv('df_part_jen.csv')
df_4.to_csv('df_part_atr.csv')
df_5.to_csv('df_part_mug.csv')
df_6.to_csv('df_part_zoe.csv')

df_pd_1 = pd.read_csv('df_part_sit.csv')
df_pd_2 = pd.read_csv('df_part_par.csv')
df_pd_3 = pd.read_csv('df_part_jen.csv')
df_pd_4 = pd.read_csv('df_part_atr.csv')
df_pd_5 = pd.read_csv('df_part_mug.csv')
df_pd_6 = pd.read_csv('df_part_zoe.csv')

df_pd_1.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
df_pd_2.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
df_pd_3.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
df_pd_4.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
df_pd_5.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
df_pd_6.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)

df_pd_1.to_csv('df_part_sittun.csv')
df_pd_2.to_csv('df_part_parth.csv')
df_pd_3.to_csv('df_part_jenna.csv')
df_pd_4.to_csv('df_part_atrima.csv')
df_pd_5.to_csv('df_part_mugundhan.csv')
df_pd_6.to_csv('df_part_zoe.csv')
