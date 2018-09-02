# coding=utf-8
from lstm.lstmmodel import get_train,get_predict

def train():
    get_train("广东省","广州市")
    get_train("上海市","上海市")
train()
def predict():
    get_predict("广东省","广州市")
    get_predict("上海市","上海市")