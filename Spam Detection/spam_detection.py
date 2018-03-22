import pandas as pd
import numpy as np

df = pd. read_csv('sms_spam_collection.csv', usecols=['v1', 'v2'])
df.columns = ['label', 'message']

# print df.head(1)

print df.describe()
print df.groupby('label').describe()

## Supervised
# predict whether a message is spam or not
# test accuracy of spam filter
# 1. Convert messages to a TF-IDF matrix where each word represents a dimension
# 2. Split dataset into a training and test set
#	 - 80/20
#	 - k cross-validation
# 3. Run classifier model on the TF-IDF matrix
# 4. Evaluate model accuracy

## Unsupervised
# find top words for spam messages
# identify topics within the spam messages