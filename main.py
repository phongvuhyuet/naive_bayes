import csv
import random
import math
import pandas

# nhom cac hang du lieu theo ket qua 
# tinh xac suat tien nghiem
priorProb = {}
def groupByTargetResult(data):
  dict = {}
  for i in range(len(data)):
    if (data[i][-1] not in dict):
      dict[data[i][-1]] = []
    dict[data[i][-1]].append(data[i])
    if (data[i][-1] not in priorProb):
      priorProb[data[i][-1]] = 1
    else:
      priorProb[data[i][-1]] +=1 
  for resultValue in priorProb:
    priorProb[resultValue] = 1.0 * priorProb[resultValue] / len(data)
  return dict

# tinh gia tri xac suat cua tat ca kha nang
def calculateAllResultProbabilities(info, test):
  probabilities = {}
  for resultValue, resultModel in info.items():
    probabilities[resultValue] = priorProb[resultValue]
    for i in range(len(resultModel)):
      mean, std_dev = resultModel[i]
      x = test[i]
      probabilities[resultValue] *= getGaussianValue(x, mean, std_dev)
  return probabilities

# dua ra du doan cho 1 bo test
def getResult(info, test):
  probabilities = calculateAllResultProbabilities(info, test)
  result, maxProbability = None, -1
  for classValue, probability in probabilities.items():
    if result is None or probability > maxProbability:
      maxProbability = probability
      result = classValue
  return result

  
# ham de chia data thanh tap train va tap test
def split_data(data, percentage):
  train_length = int(len(data) * percentage)
  test = list(data)
  train = []
  while len(train) < train_length:
    index = random.randrange(len(test))
    train.append(test.pop(index))
  return train, test



# tinh gia tri trung binh cua 1 day so
def average(numbers):
  return sum(numbers) / float(len(numbers))

# tinh do lech chuan cua 1 day so
def standardDeviation(numbers):
  avg = average(numbers)
  variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
  return math.sqrt(variance)

# tra ve trung binh va do lech chuan cua cac feature
def averageAndStandardDeviation(dataOfEach):
  info = [(average(attribute), standardDeviation(attribute)) for attribute in zip(*dataOfEach)]
  # xoa cot cuoi cung la cac gia tri phan phoi cho result chu khong phai feature
  del info[-1]
  return info

# build model: tra ve trung binh va do lech chuan cho tung gia tri ket qua
def buildModelForEachResult(data):
  model = {}
  dict = groupByTargetResult(data)
  for resultValue, dataOfEach in dict.items():
    model[resultValue] = averageAndStandardDeviation(dataOfEach)
  return model

# ham tinh gia tri ham mat do xac suat cua do lech chuan
def getGaussianValue(x, mean, stdev):
  expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
  return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo

 
# load data tu file csv 
data = csv.reader(open(r'./data.csv', "rt"))
data = list(data)

# dua ve float de tien thao tac
for i in range(len(data)):
  data[i] = [float(x) for x in data[i]]
  
# chia data thanh 2 tap: tap train va tap test
percentage = 0.8
train_data, test_data = split_data(data, percentage)

# tao model
model = buildModelForEachResult(train_data)

# test model voi bo du lieu test
results = []
for i in range(len(test_data)):
  result = getResult(model, test_data[i])
  results.append(result)
correct_count = 0
# tinh toan phan tram doan dung
for i in range(len(test_data)):
  if test_data[i][-1] == results[i]:
    correct_count += 1
rate =  (correct_count / float(len(test_data))) * 100.0

print("Do chinh xac cua thuat toan: ", rate)
