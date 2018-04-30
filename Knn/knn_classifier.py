import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Manhattan distance
def manhattan_distance(x, y):
    distance = np.abs(np.sum(x - y))
    return distance
# Euclidean distance
def euclidean_distance(x, y):
    distance = np.sqrt(np.abs(np.sum((x-y) ** 2)))
    return distance
# Sum of square distance
def sum_of_sqaure(x,y):
 	distance = np.sum(np.abs(x -y) ** 2)
 	return distance
# get batches from files function
def get_data(files, prefix=''):
    if type(files) is list:
        outcome = {}
        for file in files:
            data = {}
            # unpickle batches
            with open(prefix + file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
            for i in range(len(data[b'labels'])):
                label = data[b'labels'][i]
                if label not in outcome:
                    outcome[label] = []
                outcome[label].append(data[b'data'][i])
        return outcome
    # a file
    elif type(files) is str:
        outcome = {}
        data = {}
        with open(prefix + files, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        for i in range(len(data[b'labels'])):
            label = data[b'labels'][i]
            if label not in outcome:
                outcome[label] = []
            outcome[label].append(data[b'data'][i])
        return sort_key_dictionary(outcome)
# function that returns majority of a list
def majority_classes(data):
    maximum_item = None
    maximum_value = None
    for item in set(data):
        temp = 0
        for data_num in data:
            if item is data_num:
                temp += 1
        if maximum_value is None or maximum_value < temp:
            maximum_value = temp
            maximum_item = item
    return maximum_item
# predict function
def predict(data, test, distance):
    minimum_label = []
    minimum_label_value = []
    for index in data:
        for batch in data[index]:
            d = distance(batch, test)
            if len(minimum_label) < K:
                minimum_label.append(index)
                minimum_label_value.append(d)
            elif max(minimum_label_value) > d:
                i = np.argmax(minimum_label_value)
                minimum_label[i] = index
                minimum_label_value[i] = d
    return majority_classes(minimum_label),minimum_label
# sort dictionary by key function
def sort_key_dictionary(dictionary):
    dict_keys = list(dictionary.keys())
    dict_keys.sort()
    sorted_dictionary = {}
    for key in dict_keys:
        sorted_dictionary[key] = dictionary[key]
    return sorted_dictionary
# Hyperparameters
K = 10
D = sum_of_sqaure
if __name__ == "__main__":
    files = ['1', '2', '3', '4', '5']
    images = get_data(files, prefix='cifar-10-batches-py/data_batch_')
    test_images = get_data('test_batch', prefix='cifar-10-batches-py/') 
    result = []
    for test_image_index in test_images:
        cnt = 0
        for batch in test_images[test_image_index]:
            label, neighbor = predict(images, batch, distance=D)
            result.append(label is test_image_index)
            print("predict : %d, answer : %d"%(label, test_image_index))
            print('Possible neighbors of samples = ',neighbor)
            print("Result is",result)
            cnt += 1
            if cnt == 20:
                break          
    # print result
    mis_classify =[]
    for i in range(0,len(result)):
        if(result[i]==False):
            mis_classify.append(i)
    print("Misclassified indexes are",mis_classify)
    print("Length of miclassified samples",len(mis_classify))
    probability_error = len(mis_classify) / 200
    print("Probability of error is",probability_error)
    result_np = np.array(result, dtype='float32')
    print("Average : %f"%np.mean(result_np))

