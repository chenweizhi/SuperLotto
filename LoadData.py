#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xlrd
import numpy as np
from scipy import stats

class DataLoader:

    def __init__(self):
        self.front = None
        self.back = None
        self.sample_count = 0
        self.class_front = None
        self.class_back = None
        self.train_indices = None
        self.validation_indices= None
        self.test_indices = None
        self.test_index = 0
        self.validation_index = 0

    def get_classes_count(self):
        return list(self.class_front)[-1] + 1

    def get_seq_len(self):
        return 4

    def get_train_count(self):
        return len(self.train_indices)

    def get_test_count(self):
        return len(self.test_indices)

    def get_validation_count(self):
        return len(self.validation_indices)

    def load_xls(self, file_path):
        workbook = xlrd.open_workbook(file_path)
        #print(workbook.sheet_names())
        booksheet = workbook.sheet_by_index(0)
        front_start_idx = 9
        back_start_idx = front_start_idx + 5
        self.front = []
        self.back = []
        for idx in range(2, booksheet.nrows, 1):
            row_data = booksheet.row_values(idx)
            if row_data[front_start_idx] != '-':
                self.front.append([int(row_data[i]) for i in range(front_start_idx,front_start_idx+5,1)])
                self.back.append([int(row_data[back_start_idx]), int(row_data[back_start_idx+1])])

        list_front = []
        list_back = []
        for e in self.front:
            list_front += e
        for e in self.back:
            list_back += e
        self.sample_count = len(self.front)
        self.class_front = set(list_front)
        self.class_back = set(list_back)
        # print(self.class_front)
        # print(self.class_back)
        #spliting train set , validation set and test set
        #8 1 1
        self.train_indices = []
        self.validation_indices= []
        self.test_indices = []
        sample_index = 0
        valid_count = self.sample_count - self.sample_count % 10
        while sample_index < valid_count:
            self.train_indices += list(range(sample_index, sample_index+8, 1))
            sample_index += 8
            self.validation_indices += [sample_index]
            sample_index += 1
            self.test_indices += [sample_index]
            sample_index += 1
        # print(self.train_indices)
        # print(self.validation_indices)
        # print(self.test_indices)
        print("count {0}".format(self.sample_count))

    def get_batch(self, sample_index):
        list_input = []
        list_output = []
        for idx in sample_index:
            list_input.append(self.front[idx][0:4])
            list_output.append(self.front[idx][1:5])
        # print(list_input)
        # print(list_output)
        return np.array(list_input,dtype=np.float32), np.array(list_output)

    def next_batch_train(self, sample_count):
        sample_index = np.random.choice(self.train_indices, sample_count)
        return self.get_batch(sample_index)

    def next_batch_validation(self, sample_count):
        sample_index = list(range(self.validation_index, self.validation_index+sample_count))
        self.validation_index += sample_count
        self.validation_index = self.validation_index % len(self.validation_indices)
        sample_index = [ i % len(self.validation_indices) for i in sample_index]
        return self.get_batch(sample_index)

    def next_batch_test(self):
        sample_index = [self.test_index]
        self.test_index += 1
        self.test_index = self.test_index % len(self.test_indices)
        return self.get_batch(sample_index)

    def statistics_back(self):
        counter = [0] * (len(self.class_back)+1)
        for i in self.back:
            counter[i[0]] += 1
            counter[i[1]] += 1
        total = 0.0
        for i in counter:
            total += i
        counter = [i/total for i in counter]
        candinate = list(range((len(self.class_back)+1)))
        custm = stats.rv_discrete(name='custm', values=(candinate, counter))
        num = custm.rvs(size=2)
        return num

if __name__ == "__main__":
    print("load data")
    loader = DataLoader()
    loader.load_xls('dlt2.xls')
    loader.next_batch_train(32)
    print(loader.get_classes_count())
    loader.statistics_back()