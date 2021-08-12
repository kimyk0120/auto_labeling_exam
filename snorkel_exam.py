from utils import load_spam_dataset
import tensorflow as tf

print(tf.__version__)

print("prcs start")


# download data
DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"

df_train, df_test = load_spam_dataset()



# We pull out the label vectors for ease of use later
Y_test = df_test.label.values


print("prcs fin")