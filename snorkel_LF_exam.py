import tensorflow as tf
import glob
import zipfile
import urllib.request
import os.path
import re

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function
from snorkel.labeling import LFAnalysis
from snorkel.analysis import get_label_buckets
from snorkel.preprocess import preprocessor

from textblob import TextBlob

print(tf.__version__)

print("prcs start")


def load_spam_dataset(filenames, load_train_labels: bool = False, split_dev_valid: bool = False):
    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df = df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    df_dev = df_train.sample(100, random_state=123)

    if not load_train_labels:
        df_train["label"] = np.ones(len(df_train["label"])) * -1
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    if split_dev_valid:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_test


def download_dataset():
    DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"
    DATASET_FOLDER_ROOT = "./dataset"
    DATASET_FOLDER = '/Youtube_spam'
    urllib.request.urlretrieve(DATA_URL, DATASET_FOLDER_ROOT + '/YouTube-Spam-Collection-v1.zip')
    zip_ref = zipfile.ZipFile(DATASET_FOLDER_ROOT + '/YouTube-Spam-Collection-v1.zip', 'r')
    zip_ref.extractall(DATASET_FOLDER_ROOT + DATASET_FOLDER)
    return sorted(glob.glob(DATASET_FOLDER_ROOT + DATASET_FOLDER + "/Youtube*.csv"))


'''
ABSTAIN = -1
HAM = 0
SPAM = 1

author: Username of the comment author
date: Date and time the comment was posted
text: Raw text content of the comment
label: Whether the comment is SPAM (1), HAM (0), or UNKNOWN/ABSTAIN (-1)
video: Video the comment is associated with
'''

# download dataset
filenames = download_dataset()

# load dataset
df_train, df_test = load_spam_dataset(filenames)  # -1 로 테스트 데이터 라벨로 리턴
Y_test = df_test.label.values

sample = df_train[["author", "text", "video"]].sample(20, random_state=2)


# LF 정의
@labeling_function()
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN


@labeling_function()
def check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN


@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN


ABSTAIN = -1
HAM = 0
SPAM = 1

lfs = [check_out, check, regex_check_out]

applier = PandasLFApplier(lfs=lfs)  # LFApplier
L_train = applier.apply(df=df_train)  # L[i, j]는 j번째 레이블링 함수가 i번째 데이터 low에 대해 출력하는 레이블 -1 또는 1로 출력




# Evaluation
'''
Axis 0 will act on all the ROWS **in each COLUMN**  => 열기준
 Axis 1 will act on all the COLUMNS **in each ROW**" => 행기준
'''

coverage_check_out, coverage_check, coverage_testlf = (L_train != ABSTAIN).mean(axis=0)
print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
print(f"check coverage: {coverage_check * 100:.1f}%")
print(f"check coverage: {coverage_testlf * 100:.1f}%")



# Analysis
'''
Polarity: The set of unique labels this LF outputs (excluding abstains)
Coverage: LF 레이블 데이터 세트의 비율
Overlaps: 이 LF와 적어도 하나의 다른 LF 레이블이 있는 데이터 세트의 비율
Conflicts: 이 LF와 적어도 하나의 다른 LF 레이블이 일치하지 않는 데이터 세트의 비율
Correct: 이 LF가 올바르게 레이블을 지정하는 데이터 포인트의 수(골드 레이블이 제공된 경우)
Incorrect: 이 LF가 잘못 레이블을 지정한 데이터 포인트의 수(골드 레이블이 제공된 경우)
Empirical Accuracy: 이 LF의 경험적 정확도(골드 라벨이 제공된 경우)
'''

analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()



# check LF
'''
- 행번호(row number)로 선택하는 방법 (.iloc)
- label이나 조건표현으로 선택하는 방법 (.loc)
'''
# d) Balance accuracy and coverage
check_lf = L_train[:, 1]
check_label = df_train.iloc[check_lf == SPAM].sample(10, random_state=1)
'''
- get_label_buckets(...) to group data points by their predicted label and/or true labels.
'''
buckets = get_label_buckets(L_train[:, 1], L_train[:, 2])
check_buckets = df_train.iloc[buckets[(SPAM, ABSTAIN)]].sample(10, random_state=1)


# Writing an LF that uses a third-party model
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x


@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return HAM if x.polarity > 0.9 else ABSTAIN


@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN


lfs = [textblob_polarity, textblob_subjectivity]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

lf_anallysis = LFAnalysis(L_train, lfs).lf_summary()

# Writing More Labeling Functions
from snorkel.labeling import LabelingFunction


def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=SPAM):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


"""Spam comments talk about 'my channel', 'my video', etc."""
keyword_my = make_keyword_lf(keywords=["my"])

"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

"""Spam comments post links to other channels."""
keyword_link = make_keyword_lf(keywords=["http"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_lf(keywords=["please", "plz"])

"""Ham comments actually talk about the video's content."""
keyword_song = make_keyword_lf(keywords=["song"], label=HAM)


@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN


# LFs with Complex Preprocessors
from snorkel.preprocess.nlp import SpacyPreprocessor

# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)


@labeling_function(pre=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


from snorkel.labeling.lf.nlp import nlp_labeling_function


@nlp_labeling_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


#  Combining Labeling Function Outputs with the Label Model
lfs = [
    keyword_my,
    keyword_subscribe,
    keyword_link,
    keyword_please,
    keyword_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

lf_anallysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")


# Filtering out unlabeled data points
from snorkel.labeling import filter_unlabeled_dataframe
#
df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)



# Training a Classifier
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())



print("prcs fin")
