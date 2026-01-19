# import json

# def merge_knowledge(file1_path, file2_path):
#     try:
#         # 读取第一个 JSON 文件
#         with open(file1_path, 'r', encoding='utf-8') as file1:
#             data1 = json.load(file1)
#         # 读取第二个 JSON 文件
#         with open(file2_path, 'r', encoding='utf-8') as file2:
#             data2 = json.load(file2)

#         # 合并数据
#         for i, item in enumerate(data2):
#             if i < len(data1):
#                 item["knowledge"] = data1[i]["knowledge"]

#         # 返回合并后的数据
#         return data2

#     except FileNotFoundError:
#         print("错误: 文件未找到，请检查文件路径。")
#     except json.JSONDecodeError:
#         print("错误: JSON 数据解析失败，请检查文件格式。")
#     except Exception as e:
#         print(f"错误: 发生了一个未知错误: {e}")

#     return None

# if __name__ == "__main__":
#     file1_path = '/home/nlp/Project/XZY/JMERE-dataset/img_cor/val_processed_data.json'
#     file2_path = '/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/val_ofa_image_caption.json'
#     result = merge_knowledge(file1_path, file2_path)
#     if result is not None:
#         # 输出合并后的数据到新的 JSON 文件
#         output_file = '/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/val_ofa_image_caption_knowledge.json'
#         with open(output_file, 'w', encoding='utf-8') as outfile:
#             json.dump(result, outfile, ensure_ascii=False)
#         print(f"合并后的数据已保存到 {output_file}")
    

# import json

# def convert_json_format(input_file_path, output_file_path):
#     try:
#         # 读取输入的 JSON 文件
#         with open(input_file_path, 'r', encoding='utf-8') as infile:
#             data = json.load(infile)

#         # 打开输出文件进行写入
#         with open(output_file_path, 'w', encoding='utf-8') as outfile:
#             for item in data:
#                 # 将每个 JSON 对象转换为字符串并写入文件
#                 json_str = json.dumps(item, ensure_ascii=False)
#                 outfile.write(json_str + '\n')

#         print(f"转换完成，结果已保存到 {output_file_path}")

#     except FileNotFoundError:
#         print("错误：未找到输入文件，请检查文件路径。")
#     except json.JSONDecodeError:
#         print("错误：输入文件不是有效的 JSON 格式。")
#     except Exception as e:
#         print(f"发生未知错误：{e}")

# if __name__ == "__main__":
#     input_file = '/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/train_ofa_image_caption_knowledge.json'
#     output_file = '/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/train_ofa_knowledge.json'
#     convert_json_format(input_file, output_file)

import json
import torch
import numpy as np
from datasets import load_from_disk
import os

def merge_arrow_with_json(arrow_dir_path, json_file_path, output_file_path):
    """
    合并Arrow数据集与JSON数据
    arrow_dir_path: 包含Arrow文件的目录路径（使用load_from_disk加载）
    """
    # 初始化特征映射
    feature_map = {}
    
    # 使用load_from_disk加载Arrow数据集
    print(f"正在从 {arrow_dir_path} 加载Arrow数据集...")
    try:
        dataset = load_from_disk(arrow_dir_path)
        
        # 检查必要的字段
        required_fields = ['img_id', 'images']
        for field in required_fields:
            if field not in dataset.column_names:
                raise ValueError(f"Arrow数据集中缺少必要字段: {field}")
        
        # 提取图像特征并构建映射
        print("正在提取图像特征...")
        for example in dataset:
            img_id = example['img_id']
            image_features = example['images']
            
            # 解析特征数据
            if isinstance(image_features, str):
                # 处理字符串形式的特征列表
                import ast
                image_features = ast.literal_eval(image_features)
                image_features = torch.tensor(image_features)
            elif isinstance(image_features, np.ndarray):
                # 处理numpy数组
                image_features = torch.from_numpy(image_features)
            elif isinstance(image_features, torch.Tensor):
                # 直接使用PyTorch张量
                pass
            else:
                # 尝试转换为张量
                try:
                    image_features = torch.tensor(image_features)
                except Exception as e:
                    print(f"无法处理图像特征格式: {type(image_features)}, img_id: {img_id}")
                    continue
            
            # 转换为列表以便JSON序列化
            feature_map[img_id] = image_features.tolist()
        
        print(f"从Arrow数据集加载了 {len(feature_map)} 条图像特征")
    
    except Exception as e:
        print(f"加载Arrow数据集时出错: {e}")
        return
    
    # 处理JSON文件并写入新文件
    print("正在处理JSON文件...")
    with open(json_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        total_lines = sum(1 for _ in infile)
        infile.seek(0)  # 重置文件指针
        
        processed = 0
        matched = 0
        
        for line in infile:
            try:
                data = json.loads(line.strip())
                img_id = data.get('img_id')
                
                if img_id in feature_map:
                    # 添加图像特征到数据中
                    data['image_feature'] = feature_map[img_id]
                    matched += 1
                
                # 写入处理后的数据
                outfile.write(json.dumps(data) + '\n')
                
                processed += 1
                if processed % 100 == 0 or processed == total_lines:
                    print(f"进度: {processed}/{total_lines} ({processed/total_lines*100:.2f}%), 匹配: {matched}")
            
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line[:50]}...")
            except Exception as e:
                print(f"处理行时出错: {e}")
    
    print(f"处理完成! 总共处理了 {processed} 条记录，匹配了 {matched} 条图像特征")

if __name__ == "__main__":
    # 设置文件路径
    arrow_file_path = "/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/val_ofa_image_caption_knowledge_arrow"  # 替换为你的Arrow文件路径
    json_file_path = "/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/val_ofa_knowledge.json"    # 替换为你的JSON文件路径
    output_file_path = "/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/val_ofa_knowledge_image.json"   # 输出文件路径
    
    # 执行合并
    merge_arrow_with_json(arrow_file_path, json_file_path, output_file_path)
