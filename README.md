# Data4LLM

### The sample and useful data process tool for LLM finetuning
now including: process for json & jsonline data and output jsonlines
## Part

## API
### 1.For file
#### (1) merge files
merge all the jsonlines files with shuffle
```python
import glob
from Data4LLM import Data4LLM
files = glob("dir/*.jsonl")
Data4LLM.merge_files(files=files)
```
#### (2) split files to train and test file
```python
from Data4LLM import Data4LLM
Data4LLM.split_train_test(file_input="data/test.json",train_test_ratio=3/5)
```
### 2. For sample
#### (1) shuffle
```python
from Data4LLM import Data4LLM
Data4LLM.shuffle(file_input="data/test.jsonl")
```
#### (2) remove duplicated data
```python
from Data4LLM import Data4LLM
Data4LLM.remove_duplicate_BloomFilter(file_input="data/test.json", file_output="result/rm_dup_test.json",length=64)
```
#### (3) process property in json like rename(input->prompt, output->chosen),replace unuseful characters
```python
from Data4LLM import Data4LLM,F
def process_fn(row: dict[str:str]):
    F.replace(row, "#", "")
    F.replace(row, "https?://\S+", "")
    F.rename(row, {"input":"prompt","output":"chosen"})
    return row

Data4LLM.process_property(file_input="data/test.jsonl", file_output="result/result_test.jsonl", process_fun=process_fn)
```
#### (4) show_example 
```python
from Data4LLM import Data4LLM,F
def process_fn(row: dict[str:str]):
    F.replace(row, "#", "")
    F.replace(row, "https?://\S+", "")
    F.rename(row, {"input":"prompt","output":"chosen"})
    return row

Data4LLM.show_example(file_input="data/test.jsonl", process_fun=process_fn)
```


