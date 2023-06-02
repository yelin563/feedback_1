import streamlit as st

st.set_page_config(layout="wide")
st.title("문항 응답 예시")
st.divider()


markdown_table1 = """
| **거듭제곱의 곱셈** | **거듭제곱의 나눗셈2** | **단위의 이해** | **거듭제곱의 나눗셈1** |**수의 나눗셈** |**정오답** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 1 | 0 | 0 | 0 |
"""
markdown_table2 = """
| **등식의 성질** | **단항식의 곱셈** | **단항식의 나눗셈** | **거듭제곱의 곱셈** |**거듭제곱의 나눗셈** |**정오답** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 0 | 1 | 1 | 0 | 
"""
markdown_table3 = """
| **곱의 거듭제곱** | **거듭제곱의 나눗셈** | **다항식의 나눗셈** | **삼각형의 넓이** |**정오답** |
|  :---: | :---: | :---: | :---: | :---: | 
| 0 | 1 | 1 | 1 | 0 | 
"""
markdown_table4 = """
| **등식의 성질** | **(다항식)x(단항식)** | **단항식의 곱셈** | **(다항식)÷(단항식)** |**단항식의 나눗셈** |**정오답** |
|  :---: | :---: | :---: | :---: | :---: | :---: | -
| 0 | 1 | 1 | 0 | 0 | 0 |
"""


st.header("1-8 예시")
st.image('images/1_8_ex_2.png',  caption = "1_8_예시")
st.markdown(markdown_table1, unsafe_allow_html=True)
st.divider()
st.header("2-6 예시")
st.image('images/2_6_ex_217.png',caption = "2_6_예시")
st.markdown(markdown_table2, unsafe_allow_html=True)
st.divider()
st.header("2-7 예시")
st.image('images/2_7_ex_236.png', caption = "2_7_예시")
st.markdown(markdown_table3, unsafe_allow_html=True)
st.divider()
st.header("3-3 예시")
st.image('images/3_3_ex_12.png', caption = "3_3_예시")
st.markdown(markdown_table4, unsafe_allow_html=True)


image_path = 'images/1_8_ex_2.png'
image = open(image_path, 'rb').read()
image_width = st.image(image, caption="1_8_예시").width

# 스타일 설정
st.markdown(
    f"""
    <style>
        .image-container .image-wrapper {{
            width: {image_width}px;
        }}
        .image-container .image-wrapper img {{
            width: 100%;
        }}
        .image-container .markdown-text-container {{
            width: {image_width}px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# 이미지와 테이블 표시
st.markdown(
    f"<div class='image-container'><div class='image-wrapper'><img src='data:image/png;base64,{image}'></div>" +
    "<div class='markdown-text-container'>" +
    f"{markdown_table1}</div></div>",
    unsafe_allow_html=True
)


  

