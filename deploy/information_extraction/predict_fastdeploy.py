# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import fastdeploy
from fastdeploy.text import UIEModel, SchemaLanguage
import os
from pprint import pprint
import distutils.util


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        # required=True,
        default='deploy/information_extraction/20230221-checkpoint-10000',
        help="The directory of model, params and vocab file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument(
        "--device_id", type=int, default=0, help="device(gpu) id")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="The max length of sequence.")
    parser.add_argument(
        "--backend",
        type=str,
        default='paddle_inference',
        choices=[
            'onnx_runtime', 'paddle_inference', 'openvino', 'paddle_tensorrt',
            'tensorrt'
        ],
        help="The inference runtime backend.")
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=8,
        help="The number of threads to execute inference in cpu device.")
    parser.add_argument(
        "--use_fp16",
        type=distutils.util.strtobool,
        default=False,
        help="Use FP16 mode")
    return parser.parse_args()


def build_option(args):
    runtime_option = fastdeploy.RuntimeOption()
    # Set device
    if args.device == 'cpu':
        runtime_option.use_cpu()
        runtime_option.set_cpu_thread_num(args.cpu_num_threads)
    else:
        runtime_option.use_gpu(args.device_id)

    # Set backend
    if args.backend == 'onnx_runtime':
        runtime_option.use_ort_backend()
    elif args.backend == 'paddle_inference':
        runtime_option.use_paddle_infer_backend()
    elif args.backend == 'openvino':
        runtime_option.use_openvino_backend()
    else:
        runtime_option.use_trt_backend()
        if args.backend == 'paddle_tensorrt':
            runtime_option.enable_paddle_to_trt()
            runtime_option.enable_paddle_trt_collect_shape()
        # Only useful for single stage predict
        runtime_option.set_trt_input_shape(
            'input_ids',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        runtime_option.set_trt_input_shape(
            'token_type_ids',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        runtime_option.set_trt_input_shape(
            'pos_ids',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        runtime_option.set_trt_input_shape(
            'att_mask',
            min_shape=[1, 1],
            opt_shape=[args.batch_size, args.max_length // 2],
            max_shape=[args.batch_size, args.max_length])
        trt_file = os.path.join(args.model_dir, "inference.trt")
        if args.use_fp16:
            runtime_option.enable_trt_fp16()
            trt_file = trt_file + ".fp16"
        runtime_option.set_trt_cache_file(trt_file)
    return runtime_option


if __name__ == "__main__":
    args = parse_arguments()
    runtime_option = build_option(args)

    model_path = os.path.join(args.model_dir, "inference.pdmodel")
    param_path = os.path.join(args.model_dir, "inference.pdiparams")
    vocab_path = os.path.join(args.model_dir, "vocab.txt")

    text = ['1路贵州省人民政府驻北京办事处黔南州人民政府驻北京办事处绝密卷']
    schema = ['道段', '一级单位', '二级单位', '密级', '形状']

    t_start = time.time()
    print('开始初始化')
    uie = UIEModel(
        model_path,
        param_path,
        vocab_path,
        position_prob=0.5,
        max_length=args.max_length,
        schema=schema,
        batch_size=args.batch_size,
        runtime_option=runtime_option,
        schema_language=SchemaLanguage.ZH)
    print('初始化完成，用时: %s'%(time.time()-t_start))

    t_start = time.time()
    print("1. Named Entity Recognition Task")
    print(f"The extraction schema: {schema}")
    results = uie.predict(text, return_dict=True)
    print('预测完成，用时: %s'%(time.time()-t_start))
    pprint(results)