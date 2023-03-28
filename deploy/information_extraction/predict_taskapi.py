import time
from pprint import pprint
from paddlenlp import Taskflow


text = '1路贵州省人民政府驻北京办事处黔南州人民政府驻北京办事处绝密卷'
schema = ['道段', '一级单位', '二级单位', '密级', '形状']

t_start = time.time()
print('开始初始化')
ie = Taskflow(task='information_extraction',
              model='uie-nano',
              task_path='deploy/information_extraction/20230221-checkpoint-10000/',
#              is_static_model=True,
              device_id=-1,
              schema=schema)
t_end = time.time()
print('初始化完成，用时: %s'%(t_end-t_start))

t_start = time.time()
results = ie(text)
t_end = time.time()

print('预测完成，用时: %s'%(t_end-t_start))
pprint(results[0])