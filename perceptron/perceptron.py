def dot_product(weights, data):
    assert len(weights) == len(data)
    ret = 0
    for i in range(len(weights)):
        ret += weights[i] * int(data[i])
    return ret

def train(weights, data, label):
    data.append(1)
    f = dot_product(weights, data)

    change = False
    if f > 0 and label == '5':
        for i in range(len(weights)):
            weights[i] = weights[i] - int(data[i])
        change = True
    elif f <= 0 and label == '6':
        for i in range(len(weights)):
            weights[i] = weights[i] + int(data[i])
        change = True
    return weights, change

def main():
    f = open('../data/trainData.csv')
    train_data = f.read().split('\n')[:-1]
    f.close()

    f = open('../data/trainLabels.csv')
    train_labels = f.read().split('\n')[:-1]
    f.close()

    weights = [0 for x in range(1 + len(train_data[0].split(',')))]

    while True:
        flag = False
        for i in range(len(train_data)):
            weights, change = train(weights, train_data[i].split(','), train_labels[i])
            flag = flag or change
        if flag == False:
            break

    '''
    for i in range(len(train_data)):
        t = train_data[i].split(',')
        t.append(1)
        f = dot_product(weights, t)
        if f > 0:
            assert train_labels[i] == '6'
        if f <= 0:
            assert train_labels[i] == '5'
    '''

    f = open('../data/testData.csv')
    test_data = f.read().split('\n')[:-1]
    f.close()

    f = open('../data/testLabels.csv')
    test_labels = f.read().split('\n')[:-1]
    f.close()

    correct_count = 0
    for i in range(len(test_data)):
        test = test_data[i].split(',')
        test.append(1)
        assert len(test) == len(weights)
        dot = dot_product(weights, test)
        if (dot > 0 and test_labels[i] == '6') or (dot <= 0 and test_labels[i] == '5'):
            correct_count += 1

    print "Test accuracy:"
    print float(correct_count) / float(len(test_data))

if __name__ == "__main__":
    main()
