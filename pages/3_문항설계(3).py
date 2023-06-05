import streamlit as st

st.set_page_config(layout="wide")
st.title("문항 루브릭")
st.divider()
st.header("Part 1 거듭제곱")
st.image('images/part1_rubric.png', use_column_width=True, caption = "part1_rubric")
st.header("Part 2 단항식의 곱셈과 나눗셈, 다항식의 덧셈과 뺄셈")
st.image('images/part2_rubric.png', use_column_width=True, caption = "part2_rubric")
st.header("Part 3 단항식과 다항식의 곱셈 나눗셈")
st.image('images/part3_rubric.png', use_column_width=True, caption = "part3_rubric")

from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np
import tensorflow as tf



st.subheader("문항1-1")
st.markdown(":blue[$( - 12x^{3}y^{2} ) \div \\square \\times 18x^{3}y^{3} = 8x^{2}y^{3}$]일 때 $\square$ 안에 알맞은 식을 구하시오. ")
response = st.text_input('답안 :', "답안을 작성해주세요")

######자신의 모델에 맞는 변수 설정해주기

model_name = "1-1_kc_att_sp_70" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 70 #vocab size
emb = 100 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d = 2 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)


######model과 tokneizer 부르기

#model = RNNModel(output_d, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model = ATTModel(output_d, c) #ATTModel 쓰는경우

model.load_state_dict(torch.load("./save/"+model_name+".pt"))

######자신에게 맞는 모델로 부르기
tokenizer = AutoTokenizer.from_pretrained("./save/"+ model_name) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

######자동 채점해주는 코드

enc = tokenizer(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model(pad_ten)
label = y.squeeze().detach().cpu().numpy().round()
print("label_kc :",label)
if label[1]==1:
    st.write('정답입니다!')
else:
    st.write('오답입니다ㅜㅜ')
######자신의 모델에 맞는 변수 설정해주기

model_1_1_all = "all_kc_att_sp_170" #모델 이름 넣어주기 확장자는 넣지말기!

output_d_all = 1 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c_all = cfg(vs=170, emb=100, hidden=32, nh=4, device="cpu")


######model과 tokneizer 부르기

#model = RNNModel(output_d, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
model_all = ATTModel(output_d_all, c_all) #ATTModel 쓰는경우

model_all.load_state_dict(torch.load("./save/"+model_1_1_all+".pt"))

######자신에게 맞는 모델로 부르기
tokenizer_1_1_all = AutoTokenizer.from_pretrained("./save/"+ model_1_1_all) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

######자동 채점해주는 코드

enc_all = tokenizer_1_1_all(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc_all)
if l < max_len :
    pad_all = (max_len - l) * [0] + enc_all
else : pad_all = enc_all[l-max_len:]
pad_ten = torch.tensor(pad_all)
pad_ten = pad_ten.reshape(1,max_len)
y = model_all(pad_ten)
label_all = y.squeeze().detach().cpu().numpy().round()
print("label_all :",label)
if label_all[0]==1:
    st.write('정답입니다!')
else:
    st.write('오답입니다ㅜㅜ')
