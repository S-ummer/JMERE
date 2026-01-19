import argparse
import yaml
import torch
from attrdict import AttrDict
import pandas as pd

from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone
from src.engine import PromptTrainer, ThorTrainer
import time
import  codecs
import os
from transformers import AutoTokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Template:
    def __init__(self, args):
        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        for k, v in vars(args).items():
            setattr(config, k, v)

        set_seed(config.seed)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        # names = config.data_dir+"/"+config.model_type+"_"+config.model_size

        names = '{0}/savemodel8/{1}/{2}'.format(args.data_dir,config.reasoning,
                                                           (config.model_path+"_"+config.prompt_type+config.model_type).replace("/","_"))
        config.save_name = names
        self.config = config
        # self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        # print(self.config)
        # exit()

    # def inspect_dataloader_samples(self, dataloader, dataset_name):
    #     """查看数据加载器的前几个样本"""
    #     print(f"\n===== 查看{dataset_name}的前几个样本 =====")
    #     max_samples = 3  # 查看样本数量
        
    #     # 确保数据加载器可迭代
    #     try:
    #         dataiter = iter(dataloader)
    #     except TypeError:
    #         print(f"警告: {dataset_name} 数据加载器不可迭代")
    #         return
        
    #     # 遍历前几个样本
    #     for i in range(max_samples):
    #         try:
    #             batch = next(dataiter)
    #             print(f"\n--- {dataset_name} 样本 {i+1} ---")
                
    #             # 打印batch的基本结构
    #             print(f"Batch结构: {list(batch.keys())}")
                
    #             # 打印输入文本
    #             if 'input_ids' in batch:
    #                 # 解码输入文本
    #                 input_texts = self.tokenizer.batch_decode(
    #                     batch['input_ids'], 
    #                     skip_special_tokens=True
    #                 )
    #                 print(f"输入文本: {input_texts[0][:1000]}...")  # 截断过长文本
                    
    #             # 打印图像特征形状
    #             if 'image_features' in batch:
    #                 print(f"图像特征: {batch['image_features']}")
                    
    #             # 打印标签
    #             if 'output_ids' in batch:
    #                 output_texts = self.tokenizer.batch_decode(
    #                     batch['output_ids'], 
    #                     skip_special_tokens=True
    #                 )
    #                 print(f"目标文本: {output_texts[0]}")
                    
    #         except StopIteration:
    #             print(f"{dataset_name}样本不足 {max_samples} 个")
    #             break
    #         except Exception as e:
    #             print(f"查看样本时出错: {e}")
    #             break

            
    def forward(self):


        log_path = '{0}/log/{1}_{2}_log_result.txt'.format(args.data_dir,self.config.reasoning,
                                                           (self.config.model_path+self.config.model_size+self.config.model_type).replace("/","_"))
        
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f_out = codecs.open(log_path,'a', encoding="utf-8")


        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()

        # self.inspect_dataloader_samples(self.trainLoader, "训练集")
        # self.inspect_dataloader_samples(self.validLoader, "验证集")
        # self.inspect_dataloader_samples(self.testLoader, "测试集")

        self.model = LLMBackbone(config=self.config).to(self.config.device)

        self.config = load_params_LLM(self.config, self.model, self.trainLoader)


        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        f_out.write('\ntime:{0}\n'.format(time_str))
        arguments = " "
        for key, value in self.config.items():
            arguments += '{0}: {1} '.format(key, value)

        f_out.write(arguments)



        # print(f"Running on the {self.config.data_dir} data")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")

            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print("Choosing thor multi-step infer mode.")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)

        else:
            raise 'Should choose a correct reasoning mode: prompt or thor.'

        if self.config.zero_shot == True:
            # print("Zero-shot mode for evaluation.")
            # exit()
            r = trainer.evaluate_step(self.testLoader)
            print(r)
            return

        print("Fine-tuning mode for training.")


        best_val_str, test_str= trainer.train()


        f_out.write("\n")
        f_out.write(best_val_str)
        f_out.write("\n")
        f_out.write(test_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default='prompt', choices=['prompt', 'thor'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=False,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    parser.add_argument('-mp', '--model_path', default='google/flan-t5-base', help='model name')
    parser.add_argument('-ms', '--model_size', default='base', help='model size')
    parser.add_argument('-mt', '--model_type', default='separate', help='fusion or separate')
    parser.add_argument('-ddir', '--data_dir', default='no_none_unified_tags_txt/seed_17', help='which the few shot')
    parser.add_argument('-imf', '--image_feature_dir', default='/home/nlp/Project/XZY/JMERE-dataset/dataset/V2', help='which the image')
    parser.add_argument('-pt','--prompt_type',default="imagecap_know",choices=["imagecap","know","imagecap_know","only_text"])
    parser.add_argument('-bs','--batch_size',default=1,type=int)
    parser.add_argument('-mb', '--model_base', default="T5", type=str,choices=["T5","BART"])


    parser.add_argument('--image_feature',default=True,choices=[True,False])
    parser.add_argument('-kn', '--knowledge_number', default=1,type=int)

    args = parser.parse_args()

    template = Template(args)
    template.forward()
