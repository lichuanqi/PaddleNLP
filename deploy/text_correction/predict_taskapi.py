import time
from paddlenlp import Taskflow


text = '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。'

t_start = time.time()
print('开始初始化')
text_correction = Taskflow("text_correction")
t_end = time.time()
print('初始化完成，用时: %s'%(t_end-t_start))

t_start = time.time()
results = text_correction(text)
t_end = time.time()
print('预测完成，用时: %s'%(t_end-t_start))
print(results)