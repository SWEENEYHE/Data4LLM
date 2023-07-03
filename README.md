# JsonProcess4LLM
A JsonProcess libary for LLM data

# 1.usage
## 1.1 process jsonl data
### original data
```python
from MyDataProcess import MyDataProcess
file_in_path = "xxx.jsonl"
file_out_path = "yyy.jsonl"
#process function 处理逻辑
def deal_fn(line:dict[str,str]):
    line['input'] = line['instruction']
    line.pop('instruction')
    return line

process = MyDataProcess(file_in_path,file_out_path,deal_fn,1000)
process.show_example()
```

````
##### No 1 #####
== Before ==
{'instruction': '血热的临床表现是什么?', 'input': '', 'output': '初发或复发病不久。皮疹发展迅速，呈点滴状、钱币状或混合状。常见丘疹、斑丘疹、大小不等的斑片，潮红、鲜红或深红色。散布于体表各处或几处，以躯干、四肢多见，亦可先从头面开始，逐渐发展至全身。新皮疹不断出现，表面覆有银白色鳞屑，干燥易脱落，剥刮后有点状出血。可有同形反应;伴瘙痒、心烦口渴。大便秘结、小便短黄，舌质红赤，苔薄黄或根部黄厚，脉弦滑或滑数。血热炽盛病机，主要表现在如下四个面：一、热象：血热多属阳盛则热之实性、热性病机和病证、并表现出热象。二、血行加速：血得热则行，可使血流加速，且使脉道扩张，络脉充血，故可见面红目赤，舌色深红（即舌绛）等症。三、动血：在血行加速与脉道扩张的基础上，血分有热，可灼伤脉络，引起出血，称为“热迫血妄行”，或称动血。四、扰乱心神：血热炽盛则扰动心神，心主血脉而藏神，血脉与心相通，故血热则使心神不安，而见心烦，或躁扰发狂等症。'}
== After ==
{'input': '血热的临床表现是什么?', 'output': '初发或复发病不久。皮疹发展迅速，呈点滴状、钱币状或混合状。常见丘疹、斑丘疹、大小不等的斑片，潮红、鲜红或深红色。散布于体表各处或几处，以躯干、四肢多见，亦可先从头面开始，逐渐发展至全身。新皮疹不断出现，表面覆有银白色鳞屑，干燥易脱落，剥刮后有点状出血。可有同形反应;伴瘙痒、心烦口渴。大便秘结、小便短黄，舌质红赤，苔薄黄或根部黄厚，脉弦滑或滑数。血热炽盛病机，主要表现在如下四个面：一、热象：血热多属阳盛则热之实性、热性病机和病证、并表现出热象。二、血行加速：血得热则行，可使血流加速，且使脉道扩张，络脉充血，故可见面红目赤，舌色深红（即舌绛）等症。三、动血：在血行加速与脉道扩张的基础上，血分有热，可灼伤脉络，引起出血，称为“热迫血妄行”，或称动血。四、扰乱心神：血热炽盛则扰动心神，心主血脉而藏神，血脉与心相通，故血热则使心神不安，而见心烦，或躁扰发狂等症。'}
````
## 1.2 process json data
### original data
```
   [ {
        "instruction": "如果您是医生，请根据患者的描述回答医学问题。",
        "input": "什么是药物的 \"生物钟\" 作用？\n\n",
        "output": "生物钟是人体内一种特殊的生理节律系统，由许多细胞和分子组成，通过调节人体的代谢和生理过程来实现各种功能，包括睡眠-清醒节律、食欲、消化、内分泌和免疫反应等。\n\n药物的生物钟作用指的是一些药物可以通过调节人体内的生物钟来实现特殊的药效，例如在治疗失眠和抑郁症时，一些药物可以通过调节睡眠-清醒节律来改善患者的症状。还有一些药物可以通过调节内分泌和免疫反应来实现生物钟作用，例如调节甲状腺激素和抗体水平的药物等。生物钟作用对于人体的生理和药效具有重要意义，但也需要谨慎使用和监测不良反应。"
    },
    ....
]
```
```python
from MyDataProcess import MyDataProcess

file_in_path = "xxx.json"
file_out_path = "yyy.jsonl"


#处理逻辑
def deal_fn(line: dict[str, str]):
    line.pop('instruction')
    return line

deal = MyDataProcess(file_in_path, file_out_path, deal_fn, 1000, json=True)
deal.show_example()
deal.run()
```
```

```
