import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training

class ImageTextAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, text_embeds, image_embeds):
        # text_embeds: [batch_size, seq_len, hidden_dim]
        # image_embeds: [batch_size, hidden_dim]
        q = self.query(text_embeds)  # [batch_size, seq_len, hidden_dim]
        k = self.key(image_embeds.unsqueeze(1))  # [batch_size, 1, hidden_dim]
        v = self.value(image_embeds.unsqueeze(1))  # [batch_size, 1, hidden_dim]
        
        # 计算注意力权重
        attention_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, seq_len, 1]
        attention_weights = self.softmax(attention_scores)  # [batch_size, seq_len, 1]
        
        # 加权融合
        context = torch.matmul(attention_weights, v)  # [batch_size, seq_len, hidden_dim]
        return context

class GatedFeatureFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.image_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_features, image_features):
        """
        基于门控机制融合文本和图像特征
        text_features: [batch_size, seq_len, hidden_dim]
        image_features: [batch_size, seq_len, hidden_dim]
        """
        # 投影到相同的表示空间
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        # 拼接特征用于门控计算
        concat_features = torch.cat([text_proj, image_proj], dim=-1)  # [batch_size, seq_len, hidden_dim*2]
        
        # 计算门控值 (0-1之间)
        gate = self.gate_proj(concat_features)  # [batch_size, seq_len, 1]
        
        # 基于门控值动态融合特征
        fused_features = gate * text_proj + (1 - gate) * image_proj
        
        return fused_features

class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        self.use_lora = config.get('use_lora', True)
        self.use_int8 = config.get('use_int8', False)
        self.image_dim = 2048
        
        if self.config.model_base == "T5":
            if self.use_int8:
                self.engine = T5ForConditionalGeneration.from_pretrained(
                    config.model_path, 
                    load_in_8bit=True, 
                    device_map="auto"
                )
                self.engine = prepare_model_for_int8_training(self.engine)
            else:
                self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
        elif self.config.model_base == "BART":
            self.engine = BartForConditionalGeneration.from_pretrained(config.model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        # 获取模型隐藏层维度
        self.hidden_dim = self.engine.config.hidden_size
        
        # 初始化图像特征处理组件
        self.image_projection = nn.Linear(self.image_dim, self.hidden_dim)
        self.image_attention = ImageTextAttention(self.hidden_dim)
        
        # 门控融合机制
        self.gated_fusion = GatedFeatureFusion(self.hidden_dim)
        
        # 可选：添加融合后的特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        if self.use_lora:
            self.setup_lora()

    def setup_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.config.get('lora_r', 8),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            bias="none",
            target_modules=self._get_lora_target_modules()
        )
        self.engine = get_peft_model(self.engine, lora_config)
        self.engine.print_trainable_parameters()

    def _get_lora_target_modules(self):
        if self.config.model_base == "T5":
            return ["q", "v"]
        elif self.config.model_base == "BART":
            return ["q_proj", "v_proj"]
        return []

    def forward(self, **kwargs):
        input_ids = kwargs.get('input_ids')
        input_masks = kwargs.get('input_masks')
        output_ids = kwargs.get('output_ids')
        output_masks = kwargs.get('output_masks')
        image_features = kwargs.get('image_features')  # [batch_size, image_dim]
        
        if output_ids is not None:
            output_ids = output_ids.clone()
            output_ids[output_ids == self.tokenizer.pad_token_id] = -100
        
        # 获取文本嵌入
        inputs_embeds = self.engine.get_input_embeddings()(input_ids)  # [batch_size, seq_len, hidden_dim]
        
        # 处理图像特征
        if image_features is not None:
            # 投影图像特征到文本嵌入空间
            projected_image = self.image_projection(image_features)  # [batch_size, hidden_dim]
            
            # 使用注意力机制融合图像和文本
            image_context = self.image_attention(inputs_embeds, projected_image)
            
            # 使用门控机制融合文本和图像特征
            fused_embeds = self.gated_fusion(inputs_embeds, image_context)
            
            # 可选：特征增强
            fused_embeds = self.feature_enhancer(fused_embeds)
            
            # 使用融合后的嵌入作为模型输入
            inputs_embeds = fused_embeds
        
        # 使用处理后的嵌入作为输入
        output = self.engine(
            inputs_embeds=inputs_embeds,  # 使用融合后的嵌入而非 input_ids
            attention_mask=input_masks,
            decoder_input_ids=None,
            decoder_attention_mask=output_masks,
            labels=output_ids
        )
        return output.loss

    def generate(self, input_ids, input_masks, image_features=None, **kwargs):
        # 获取文本嵌入
        inputs_embeds = self.engine.get_input_embeddings()(input_ids)
        
        # 处理图像特征（与 forward 方法类似）
        if image_features is not None:
            projected_image = self.image_projection(image_features)
            image_context = self.image_attention(inputs_embeds, projected_image)
            fused_embeds = self.gated_fusion(inputs_embeds, image_context)
            inputs_embeds = self.feature_enhancer(fused_embeds)
        
        # 使用处理后的嵌入进行生成
        generation_kwargs = {
            'max_length': self.config.get('max_length', 512),
            **kwargs
        }
        
        output = self.engine.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=input_masks,
            **generation_kwargs
        )
        
        dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
        return [text.strip() for text in dec]

    def evaluate(self, input_ids, input_masks, image_features=None):
        output = self.generate(input_ids, input_masks, image_features, max_length=300)
        label_dict = {w: i for i, w in enumerate(self.config.get('label_list', []))}
        return [label_dict.get(text, 0) for text in output]

    def test_to_word(self, input_ids):
        dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        return [text.strip() for text in dec]

    def save_pretrained(self, save_directory):
        if self.use_lora:
            self.engine.save_pretrained(save_directory)
        else:
            super().save_pretrained(save_directory)