import torch
import clip

def text_prompt(data, config):
    if config.network.use_text_prompt_learning:
        # 将类别名称中的空格替换为下划线并存储到 classname 中
        classname = [c.replace(' ', '_') for _, c in data.classes]
        
        # 打印 classname 验证结果
        # print(classname)
        return classname, len(classname)
    else:
        # text_aug = ['{}']
        text_aug = 'This is a video about {}'
        # for i, c in data.classes:
        #     if i <=2:
        #         print(text_aug.format(c))
        # print("data.classes:", data.classes)
        classes = torch.cat([clip.tokenize(text_aug.format(c)) for i, c in data.classes])
        return classes, classes.size(0)