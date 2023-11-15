# -*- encoding: utf-8 -*-
"""

@File    :   config.py  
@Modify Time : 2023/11/15 20:36 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 描述命令行读取过程大体

"""
import argparse

def parse_arguments():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Machine Translation Training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--test_interval', type=int, default=5, help='test interval')
    parser.add_argument('--resume_model', type=str, default='', help='path to resume model')
    return parser.parse_args()