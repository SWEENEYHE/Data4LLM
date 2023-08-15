import logging
import os.path
import random
import re

import jsonlines
import pandas as pd
from tqdm import tqdm
import traceback
import jieba
from simhash import Simhash


def setLogger(fileName):
    # 配置日志记录器
    logger = logging.getLogger(fileName)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(fileName)
    fh.setLevel(logging.DEBUG)

    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)

    formater = logging.Formatter("%(message)s")
    fh.setFormatter(formater)
    # ch.setFormatter(formater)

    logger.addHandler(fh)
    # logger.addHandler(ch)
    return logger


'''
    Data Process class for json & jsonline to generate data for LLM, including:
    1.For row (sample)
        1.1.process every row (rename property, remove extra property, add property...)
        1.2.shuffle rows
        1.3 remove duplicated rows
        1.4 split train and test set
    2.For file
        2.1 transfer json to jsonline
        2.2 merge jsonline files
'''


class SFT:

    # 封装保存过程
    @classmethod
    def __save_and_clear__(cls, writer, items, close=False):
        writer.write_all(items)
        items.clear()
        if close:
            writer.close()

    '''
    pre prorcess
    '''

    @classmethod
    def __preprocess__(cls, file_input, process_fun, json=None):
        postfix = os.path.splitext(file_input)
        if json is None:
            json = postfix == "json"
            # 如果是json
        if json:
            reader = pd.read_json(file_input).iterrows()
            # 预处理函数
            pre_process = lambda x: x[1].to_dict()
            # 由于iterrows()结果为（index,Series),包装一个适配器将Series取出并转换成dict，确保处理统一
            process_fun = lambda x: process_fun(pre_process(x))
            # 如果是jsonline
        else:
            reader = jsonlines.open(file_input)
            pre_process = lambda x: x
        return reader, pre_process, process_fun

    @classmethod
    def __get_simhash__(cls, text, stopwords_file="stopwords.txt", length=64):
        path = os.path.abspath(__file__)
        dir = os.path.dirname(path)
        stopwords_file = dir + os.sep + stopwords_file
        # 分词和停用词
        stopwords = set()
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
        words = [word for word in jieba.cut(text) if word not in stopwords]
        words = " ".join(words)
        simhash = Simhash(words, f=length)
        return simhash

    # 计算所有样本的simhash值，存储到文件中
    @classmethod
    def __simhash_all_rows__(cls, file_input, file_output=None, max_row_limit=1000, length=64):
        '''
        __simhash_all_rows__ : generate all the simhash value of the intput file and cache them to file
        :param file_input: input file path
        :param max_row_limit: the max data number in memory which is useful to save memory
        :param length: the simhash length
        :return:
        '''
        print("1.hash all the rows")
        if file_output is None:
            file_output = file_input.split(".")[0] + "_hash.jsonl"
        reader = jsonlines.open(file_input)
        writer = jsonlines.open(file_output, mode="w")
        write_buffer = []

        i = 0
        for row in tqdm(reader, position=0):
            row_text = " ".join(row.values())
            hash_row = cls.__get_simhash__(row_text, length=length).value
            row['hash'] = hash_row
            write_buffer.append(row)

            i += 1
            if i % max_row_limit == 0:
                cls.__save_and_clear__(writer, write_buffer)
        cls.__save_and_clear__(writer, write_buffer, close=True)
        return i

    '''
        1. For row(sample)
    '''

    @classmethod
    def process_property(cls, file_input, file_output, process_fun, max_row_limit=1000, json=None):
        '''
        process_property: process the json row one by one, including: rename property, remove property, process content(remove chars, replace chars)
        :param file_input: input file path
        :param file_output: output file path
        :param process_fun: process function
        :param max_row_limit: default=1000, every step to write file and max data num in memory
        :param json: default=None, it determines json or jsonline, or True/False
        '''
        reader, _, process_fun = cls.__preprocess__(file_input, process_fun, json)
        writer = jsonlines.open(file_output, mode="a")

        # 输出文件存在且大于0字节说明已经有内容，提示先删除 ： 由于进行追加，重复生成时导致非预期的结果，且很难排查（每次打开文件查看只看top k行以为新修改的逻辑没生效）
        assert1 = os.path.exists(file_output)
        assert2 = os.path.getsize(file_output) == 0
        assert (not assert1) or (
                assert1 and assert2), f"file_out_path {file_output} exists and is not empty, please remove it ahead!"

        items = []
        i = 0
        for item in tqdm(reader, position=0):
            try:
                item = process_fun(item)
                if item is not None:
                    items.append(item)
                # 达到最大内存处理数，则写入文件
                i += 1
                if i % max_row_limit == 0:
                    cls.__save_and_clear__(writer, items)
            except Exception as e:
                cls.__save_and_clear__(writer, items)
                print(f"Error: encounter an error when dealing with the row {i + 1} : {e}")
                traceback.print_exception(type(e), e, e.__traceback__)
                break
        # 最后一次写文件
        cls.__save_and_clear__(writer, items, close=True)

    @classmethod
    def show_example(cls, file_input, process_fun, json=None, s=0, e=5):
        '''
        
        :param file_input: 
        :param process_fun: 
        :param json: 
        :param s: 
        :param e: 
        :return: 
        '''
        reader, preprocess_fun, process_fun = cls.__preprocess__(file_input, process_fun, json)
        assert s <= e, f"s should >= e, but got s={s} <= e= {e}"
        assert s >= 0, f"s should >= 0 , but got s={s}"

        i = 0
        for item in reader:
            i += 1
            if i >= e:
                break
            if i <= s:
                continue
            print(f"##### No {i} #####")
            print(f"== Before ==\n{preprocess_fun(item)}")
            print(f"== After ==\n{process_fun(item)}")

    # shuffle
    @classmethod
    def shuffle(cls, file_input, file_output):
        '''
        shuffle: shuffle all the data in input file. warning: it loads all the data in memory
        :param file_input: input data
        :param file_output: output data
        :return:
        '''
        reader = jsonlines.open(file_input)
        writer = jsonlines.open(file_output, mode="w")
        buffer = []
        for row in reader:
            buffer.append(row)

        # shuffle
        random.shuffle(buffer)
        writer.write_all(buffer)
        writer.close()

    '''
        1.3 remove duplicate data
    '''

    # 基于simhash比较去重（需要两两比较，时间复杂度高，不适合百万级别以上数据）
    @classmethod
    def remove_duplicate(cls, file_input, file_output, ratio=1, max_row_limit=1000, skip_hash=False, length=64,
                         log_path="result.log"):
        '''
            remove_duplicate : remove duplicate data by sim_hash, which compares data one by one, getting more accurate and finely result but costing massive time
            :param file_input: input file path with duplicated data
            :param file_output: result file path
            :param ratio: threshold for duplication, which is actually the distance of the two simhash value
            :param max_row_limit: the max data number in memory which is useful to save memory
            :param skip_hash: default false. it needed when call the function in first time, which is used to get the simhash in all the data
            :param length: the simhash length
            :param log_path: log file path
            :return: result data number , removed data number
        '''
        logger = setLogger(log_path)
        pbar = tqdm(position=0)
        if not skip_hash:
            # 1.hash all the rows
            total = cls.__simhash_all_rows__(file_input, None, max_row_limit, length=length)
            pbar = tqdm(total=total, position=0)

        file_hash = file_input.split(".")[0] + "_hash.jsonl"

        # 2.get all the distance
        print("2.get all the distance")
        reader1 = jsonlines.open(file_hash)
        writer = jsonlines.open(file_output, mode="w")
        rm_set = set()
        write_buffer = []
        len_total = 0
        len_rm = 0
        # 遍历第i个

        for i, row1 in enumerate(reader1):
            len_total += 1
            # 如果第i个已经在移除集了，则无需计算
            if i in rm_set:
                continue

            # 将第i个加入写列表
            row1_hash = row1['hash']
            row1.pop('hash')
            write_buffer.append(row1)

            if i % max_row_limit == 0:
                cls.__save_and_clear__(writer, write_buffer)

            # 遍历i+1，计算距离
            reader2 = jsonlines.open(file_hash)
            for j, row2 in enumerate(reader2):
                if j <= i:
                    continue
                simhash1 = Simhash(row1_hash, f=length)
                simhash2 = Simhash(row2['hash'], f=length)
                distance = simhash1.distance(simhash2)
                if distance < ratio:
                    logger.log(level=logging.INFO, msg=f"or_row:{row1} \nrm_row:{row2}\ndis:{distance}\n====\n")
                    rm_set.add(j)
                    len_rm += 1

            reader2.close()
            pbar.update(1)
            pbar.set_postfix(total=len_total, rm=len_rm)

        cls.__save_and_clear__(writer, write_buffer, close=True)
        # 删除临时的hash文件
        if os.path.exists(file_hash):
            os.remove(file_hash)
        return len_total, len_rm

    # 基于simhash做布尔筛去重（粗筛，速度快，适合百万级别以上数据）
    @classmethod
    def remove_duplicate_BloomFilter(cls, file_input, file_output, max_row_limit=1000, skip_hash=False, length=64,
                                     log_path="result.log"):
        '''
            remove_duplicate : remove duplicate data by sim_hash, which removes data by bloom filter, very fast
            :param file_input: input file path with duplicated data
            :param file_output: result file path
            :param max_row_limit: the max data number in memory which is useful to save memory
            :param skip_hash: default false. it needed when call the function in first time, which is used to get the simhash in all the data
            :param length: the simhash length
            :param log_path: log file path
            :return: result data number , removed data number
        '''
        logger = setLogger(log_path)
        pbar = tqdm(position=0)
        if not skip_hash:
            # 1.hash all the rows
            total = cls.__simhash_all_rows__(file_input, None, max_row_limit, length=length)
            pbar = tqdm(total=total, position=0)

        file_hash = file_input.split(".")[0] + "_hash.jsonl"

        # 2.get all the distance
        print("2.get all the distance")
        reader1 = jsonlines.open(file_hash)
        writer = jsonlines.open(file_output, mode="w")
        keep_dict = {}
        write_buffer = []
        len_total = 0
        len_rm = 0
        # 遍历第i个
        for i, row in enumerate(reader1):
            len_total += 1
            # 如果第i个hash已经在集合了，则无需计算
            if row['hash'] in keep_dict:
                len_rm += 1
                logger.log(level=logging.INFO, msg=f"{i}\nor_row:{keep_dict[row['hash']]}\nrm_row:{row}")
                continue

            # 加入集合
            keep_dict[row['hash']] = row
            row.pop('hash')
            # 将第i个加入写列表
            write_buffer.append(row)

            if i % max_row_limit == 0:
                cls.__save_and_clear__(writer, write_buffer)

            pbar.update(1)
            pbar.set_postfix(total=len_total, rm=len_rm)

        cls.__save_and_clear__(writer, write_buffer, close=True)
        # 删除临时的hash文件
        if os.path.exists(file_hash):
            os.remove(file_hash)
        return len_total, len_rm

    '''

    2.For file

    '''

    @classmethod
    def merge_files(cls, files: list[str], file_output="merge.jsonl", shuffle=True, max_row_limit=1000):
        '''
        merge_files: merge all the files data, if
        :param file_output:
        :param files:
        :param shuffle:
        :param max_row_limit:
        :return:
        '''
        writer = jsonlines.open(file_output, mode="w")

        if not shuffle:
            for file in files:
                reader = jsonlines.open(file)
                buffer = []
                # 遍历行
                for i, row in enumerate(reader):
                    if i % max_row_limit == 0:
                        cls.__save_and_clear__(writer, buffer)
                    buffer.append(row)
            # 结束
            cls.__save_and_clear__(writer, buffer, close=True)
        else:
            rows = []
            # 同时打开多个文件
            readers = [jsonlines.open(file) for file in files]
            pbar = tqdm(position=0)
            # readers池不为空，则持续循环
            while len(readers) > 0:
                for reader in readers:
                    try:
                        # 轮流提取行
                        row = reader.read()
                        rows.append(row)
                        # 每max_row_limit条数据就写一次磁盘
                        if len(rows) % max_row_limit == 0:
                            cls.__save_and_clear__(writer, rows)
                        pbar.update()
                    except EOFError:
                        # EOF表明到达终点，从池子里移除该reader
                        readers.remove(reader)
            # 保存并关闭
            cls.__save_and_clear__(writer, rows, close=True)
            # shuffle
            cls.shuffle(file_output, file_output)

    @classmethod
    def split_train_test(cls, file_input, train_test_ratio, file_train_output=None, file_test_output=None):
        '''
        split_train_test: split input file data to train set and test set by the train_test_ratio
        :param file_input: input file path
        :param train_test_ratio: train_test_ration=train number / total number = train_number / (train number+ test number)
        :param file_train_output: default file_input.split(".")[0]+"_train.jsonl"
        :param file_test_output: default file_input.split(".")[0]+"_test.jsonl"
        :return: None
        '''
        file_base_name = file_input.split(".")[0]
        if file_train_output is None:
            file_train_output = file_base_name + "_train.jsonl"
            file_test_output = file_base_name + "_test.jsonl"

        reader = jsonlines.open(file_input)
        test_writer = jsonlines.open(file_train_output, mode="w")
        train_writer = jsonlines.open(file_test_output, mode="w")

        buffer = []
        for row in reader:
            buffer.append(row)

        row_num = len(buffer)
        # shuffle
        random.shuffle(buffer)
        # count length
        train_len = int(row_num * train_test_ratio)
        # split
        train_buffer = buffer[:train_len]
        test_buffer = buffer[train_len:]
        # shuffle
        random.shuffle(train_buffer)
        random.shuffle(test_buffer)
        # save and close
        cls.__save_and_clear__(train_writer, train_buffer, close=True)
        cls.__save_and_clear__(test_writer, test_buffer, close=True)


class F:
    @classmethod
    def rename(cls, row, mapping: dict[str:str]) -> None:
        for old_k, new_k in mapping.items():
            if old_k in row:
                row[new_k] = row.pop(old_k)

    @classmethod
    def replace(cls, row, pattern, repl, property=None) -> None:
        if property is None:
            property = row.keys()

        for k in property:
            row[k] = re.sub(pattern, repl, row[k])
    @classmethod
    def get_length(cls, row, property=None) -> int:
        '''
        get length of the json
        :param row:
        :param property:
        :return:
        '''
        if property is None:
            item = "".join(row.values())
        else:
            item = ""
            for k, v in row.items():
                if k in property:
                    item += v
        return len(item)
    @classmethod
    def get_count(cls, file_input):
        """
        get the sample number of a file
        :param file_input:
        :return:
        """

        _, postfix = os.path.splitext(file_input)
        allow_postfix = {".jsonl", ".json", ".txt"}
        assert postfix in allow_postfix, f"The postfix  {postfix} is not supported, expect {allow_postfix}"

        if postfix == ".jsonl":
            reader = jsonlines.open(file_input)
        elif postfix == ".json":
            reader = pd.read_json(file_input).iterrows()
        elif postfix == ".txt":
            reader = open(file_input, mode="r")

        num = 0
        for row in reader:
            num += 1
        reader.close()
        return num

class PT:
    @classmethod
    def show_properties(cls, files, s=0, e=5):
        '''
        show the json structure
        :param files:
        :param s:
        :param e:
        :return:
        '''
        for i, file in enumerate(files):
            if i < s:
                continue
            elif i > e:
                break
            print(f"====={i}=====")
            reader = pd.read_json(file)
            dic = reader.to_dict()
            print(dic)

    @classmethod
    def parse_pages(cls, files, process_fun, output_dir):
        '''
        parse the semi structure json and parse all the token needed together fot PT
        :param files:
        :param process_fun:
        :param output_dir:
        :return:
        '''
        for file in tqdm(files):
            try:
                # 读取数据
                reader = pd.read_json(file)
                dic = reader.to_dict()
                # 处理数据
                result = process_fun(dic)
                # 写文件
                path, filename = os.path.split(file)
                filename, _ = os.path.splitext(filename)
                dir_filename = os.path.join(output_dir, filename)
                with open(dir_filename + ".txt", mode='w') as f:
                    f.write(result)
            except Exception as e:
                print(f"====={file}======")
                traceback.print_exception(type(e), e, e.__traceback__)
                print(e.args)

    @classmethod
    def merge_files(cls, files, output_file="merge_file.txt", max_limit_num=100):
        '''
        merge all the txt files
        :param files:
        :param output_file:
        :param max_limit_num:
        :return:
        '''
        # 按长度切分列表
        def split_list_by_length(lst, length):
            return [lst[i:i + length] for i in range(0, len(lst), length)]

        split_files = split_list_by_length(files, max_limit_num)
        with open(output_file, mode="a") as writer:
            for split_file in tqdm(split_files):
                contents = []
                for file in split_file:
                    with open(file, mode="r") as reader:
                        content = reader.readlines()
                    contents += content
                random.shuffle(contents)
                writer.writelines(contents)

    @classmethod
    def split_train_test(cls, file_input, train_test_ratio, file_train_output="train.txt", file_test_output="test.txt"):
        '''
        split a file into train and test files
        :param file_input:
        :param train_test_ratio:
        :param file_train_output:
        :param file_test_output:
        :return:
        '''
        with open(file_input, mode="r") as reader:
            buffer = reader.readlines()

        row_num = len(buffer)
        # shuffle
        random.shuffle(buffer)
        # count length
        train_len = int(row_num * train_test_ratio)
        # split
        train_buffer = buffer[:train_len]
        test_buffer = buffer[train_len:]
        # shuffle
        random.shuffle(train_buffer)
        random.shuffle(test_buffer)

        with open(file_train_output, mode="w") as writer1:
            writer1.writelines(train_buffer)

        with open(file_test_output, mode="w") as writer2:
            writer2.writelines(test_buffer)

