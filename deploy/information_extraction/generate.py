# 数据生成
import random

MIJI = ['机密','秘密','绝密']
XINGZHUANG = ['信', '包', '卷', '箱', '袋']

text1 = ['北京工业大学']
text2 = ['研究生招生办公室']

for text1_ in text1:
    for text2_ in text2:
        for num in range(10):

            n1 = random.randint(1, 10)
            n2 = random.randint(0, 2)
            n3 = random.randint(0, 4)
            duan = f'{n1}段'
            miji = MIJI[n2]
            xingzhuang = XINGZHUANG[n3]

            text_new = f'{duan}{text1_}{text2_}{miji}{xingzhuang}'
            print(text_new)