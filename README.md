# Data4LLM
<div align="center"> 
<a href="https://github.com/SWEENEYHE/Data4LLM/blob/main/LICENSE.txt">
<img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-green">
</a>
<a href="https://pypi.org/project/data4llm/0.1.1/">
<img alt="Static Badge" src="https://img.shields.io/badge/pypi-0.1.1-blue">
</a>
</div>

### The sample and useful data process tool for LLM finetuning now including: process for json & jsonline data and output jsonlines
### it runs well in million number level
# install
```
pip install data4llm
```
# API
## SFT
```python
from data4llm.Data4LLM import SFT
```

### 1.For file level
#### (1) merge files
merge all the jsonlines files with shuffle
```python
import glob
from data4llm import Data4LLM

files = glob("dir/*.jsonl")
Data4LLM.merge_files(files=files)
```
#### (2) split files to train and test file

```python
from data4llm.Data4LLM import SFT

SFT.split_train_test(file_input="data/test.json", train_test_ratio=3 / 5)
```
### 2. For sample level
Every sample is a json with key-value form dict[str:str],like
````
 {"input":"hello!","output":"Hi, I'm an AI assistant, how can I help you?"}
````
#### (1) shuffle
shuffle all the json in a file, it doesn't optimize the memory usage now, requiring to load all the data to memory

```python
from data4llm.Data4LLM import SFT

SFT.shuffle(file_input="data/test.txt", file_output="result/sh_test.jsonl")
```
````
def shuffle(cls, file_input, file_output):
    shuffle: shuffle all the data in input file. warning: it loads all the data in memory
    ile_input: input file path
    file_output: output file path
````

#### (2) remove duplicated data
remove duplicate data by sim_hash. There are two function `remove_duplicate_BloomFilter` and `remove_duplicate`.

`remove_duplicate` : remove duplicate data by sim_hash, which removes data by bloom filter, very fast

```python
from data4llm.Data4LLM import SFT
SFT.remove_duplicate_BloomFilter(file_input="data/test.json", file_output="result/rm_dup_test.json", length=64)
```
````
def remove_duplicate_BloomFilter(cls, file_input, file_output, max_row_limit=1000, skip_hash=False, length=64,
                                 log_path="result.log"):
    '''
        remove_duplicate : remove duplicate data by sim_hash, which removes data by bloom filter, very fast
        file_input: input file path with duplicated data
        file_output: result file path
        max_row_limit: the max data number in memory which is useful to save memory
        skip_hash: default false. it needed when call the function in first time, which is used to get the simhash in all the data
        length: the simhash length
        log_path: log file path
        :return: result data number , removed data number
    '''
````
`remove_duplicate` : remove duplicate data by sim_hash, which compares data one by one, getting more accurate and finely result but costing massive time

```python
from data4llm.Data4LLM import SFT

SFT.remove_duplicate(file_input="data/test.json", file_output="result/rm_dup_test.json", length=64)
```
````

def remove_duplicate(cls, file_input, file_output, ratio=1, max_row_limit=1000, skip_hash=False, length=64,
                 log_path="result.log"):
    remove_duplicate : remove duplicate data by sim_hash, which compares data one by one, getting more accurate and finely result but costing massive time
    file_input: input file path with duplicated data
    file_output: result file path
    ratio: threshold for duplication, which is actually the distance of the two simhash value
    max_row_limit: the max data number in memory which is useful to save memory
    skip_hash: default false. it needed when call the function in first time, which is used to get the simhash in all the data
    length: the simhash length
    log_path: log file path
    :return: result data number , removed data number
````

#### (3) process property in json
process the json row one by one, including: rename property, remove property, process content(remove chars, replace chars)

```python
from data4llm.Data4LLM import SFT, F


# define a process function to process every json row
def process_fn(row: dict[str:str]):
    '''
        row is a json in dict[str:str] form, you can process it with dict function by yourself, we also define some useful functions in Data2LLM.F
        replace chars
    '''
    # details in F section
    F.replace(row, "#", "")   # use regrex to replace all the '#' to '' / remove all the '#'
    F.replace(row, "https?://\S+", "")  # use reg to remove url
    '''
        rename chas
        rename json property ,'input' to 'prompt', 'output' to 'chosen'
        {"input":"hello!","output":"Hi, I'm an AI assistant, how can I help you?"}=>{"prompt":"hello!","chosen":"Hi, I'm an AI assistant, how can I help you?"}
    '''
    F.rename(row, {"input": "prompt", "output": "chosen"})
    '''
        you can also process the row: dict[str:str] by yourself:
        row['key']='value'
        row['key'] = row.pop('key1')+row.pop('key2')
        ...
    '''
    return row


SFT.process_property(file_input="data/test.txt", file_output="result/result_test.jsonl", process_fun=process_fn)
```

````
def process_property(cls, file_input, file_output, process_fun, max_row_limit=1000, json=None):
    process_property: process the json row one by one, including: rename property, remove property, process content(remove chars, replace chars)
     file_input: input file path
     file_output: output file path
     process_fun: process function
     max_row_limit: default=1000, every step to write file and max data num in memory
     json: default=None, it determines json or jsonline, or True/False
````


#### (4) show_example
it is very useful to show the result before actually conduct by using show_example:
```python
from data4llm.Data4LLM import SFT

SFT.show_example(file_input="data/test.txt", process_fun=process_fn)
```
examples:
````
##### No 1 #####
== Before ==
{'input': 'welcome to https://www.baidu.com #LLM world', 'output': 'I like #LLM'}
== After ==
{'prompt': 'welcome to  LLM world', 'chosen': 'I like LLM'}
##### No 2 #####
== Before ==
{'input': 'hello!', 'output': "Hi, I'm an AI assistant, how can I help you?"}
== After ==
{'prompt': 'hello!', 'chosen': "Hi, I'm an AI assistant, how can I help you?"}
````
```
def show_example(cls, file_input, process_fun, json=None, s=0, e=5):
    file_input: 
    process_fun: 
    json: if the file is json or jsonline, default None means it decided by the postfix of th file_input 
    s: default 0 the start row num
    e: default 5 the end row num
    :return: None
```

## PT
```python
from data4llm.Data4LLM import PT
```

#### (1)  show_properties
show the json structure
```python
def show_properties(cls, files, s=0, e=5):
        '''
        show the json structure
        :param files:
        :param s:
        :param e:
        :return:
        '''
```

#### (2) parse_pages
parse the semi structure json and parse all the token needed together fot PT
```python
def parse_pages(cls, files, process_fun, output_dir):
        '''
        parse the semi structure json and parse all the token needed together fot PT
        :param files:
        :param process_fun:
        :param output_dir:
        :return:
        '''
```
### (3) merge_files
  merge all the txt files
````python
def merge_files(cls, files, output_file="merge_file.txt", max_limit_num=100):
    '''
    merge all the txt files
    :param files: 
    :param output_file: 
    :param max_limit_num: 
    :return: 
    '''
````
### (4) split_train_test
split a file into train and test files
```python 
def split_train_test(cls, file_input, train_test_ratio, file_train_output="train.txt", file_test_output="test.txt"):
    '''
    split a file into train and test files
    :param file_input: 
    :param train_test_ratio: 
    :param file_train_output: 
    :param file_test_output: 
    :return: 
    '''
```

## F
A util class offering some useful functions
```python
from data4llm.Data4LLM import F
```
### (1) getSize
get the sample number of a file
```python
def getSize(cls, file_input):
    """
    get the sample number of a file
    :param file_input:
    :return:
    """
```

### (2) property process function in SFT
`rename()` : rename the property of every json \
`repalce()`: replace the chars in a json or a property in the json
```python
def rename(cls, row, mapping: dict[str:str]) -> None
def replace(cls, row, pattern, repl, property=None) -> None
```