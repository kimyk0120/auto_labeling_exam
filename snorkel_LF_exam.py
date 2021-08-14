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


print("prcs fin")
