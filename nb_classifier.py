# import the necessary packages
from PIL import Image
import numpy as np
import argparse
import os, sys
import random
import math
import cv2 as cv

data = []
files = []
labels = []
totalPredictions = []
objToIndexDict = {}

objectDict = {
	"001": "ak47",
	"002": "american-flag",
	"003": "backpack",
	"004": "baseball-bat",
	"005": "baseball-glove",
	"006": "basketball-hoop",
	"007": "bat",
	"008": "bathtub",
	"009": "bear",
	"010": "beer-mug",
	"011": "billiards",
	"012": "binoculars",
	"013": "birdbath",
	"014": "blimp",
	"015": "bonsai-101",
	"016": "boom-box",
	"017": "bowling-ball", 
	"018": "bowling-pin",
	"019": "boxing-glove",
	"020": "brain-101",
	"021": "breadmaker",
	"022": "buddha-101", 
	"023": "bulldozer",
	"024": "butterfly",
	"025": "cactus",
	"026": "cake",
	"027": "calculator",
	"028": "camel",
	"029": "cannon",
	"030": "canoe",
	"031": "car-tire",
	"032": "cartman",
	"033": "cd",
	"034": "centipede",
	"035": "cereal-box",
	"036": "chandelier-101",
	"037": "chess-board",
	"038": "chimp",
	"039": "chopsticks",
	"040": "cockroach",
	"041": "coffee-mug",
	"042": "coffin",
	"043": "coin",
	"044": "comet",
	"045": "computer-keyboard",
	"046": "computer-monitor",
	"047": "computer-mouse",
	"048": "conch",
	"049": "cormorant",
	"050": "covered-wagon",
	"051": "cowboy-hat",
	"052": "crab-101",
	"053": "desk-globe",
	"054": "diamond-ring",
	"055": "dice",
	"056": "dog",
	"057": "dolphin-101",
	"058": "doorknob",
	"059": "drinking-straw",
	"060": "duck",
	"061": "dumb-bell",
	"062": "eiffel-tower",
	"063": "electric-guitar-101",
	"064": "elephant-101",
	"065": "elk",
	"066": "ewer-101",
	"067": "eyeglasses",
	"068": "fern",
	"069": "fighter-jet",
	"070": "fire-extinguisher",
	"071": "fire-hydrant",
	"072": "fire-truck",
	"073": "fireworks",
	"074": "flashlight",
	"075": "floppy-disk",
	"076": "football-helmet",
	"077": "french-horn",
	"078": "fried-eggs",
	"079": "frisbee",
	"080": "frog",
	"081": "frying-pan",
	"082": "galaxy",
	"083": "gas-pump",
	"084": "giraffe",
	"085": "goat",
	"086": "golden-gate-bridge",
	"087": "goldfish",
	"088": "golf-ball",
	"089": "goose",
	"090": "gorilla",
	"091": "grand-piano-101",
	"092": "grapes",
	"093": "grasshopper",
	"094": "guitar-pick",
	"095": "hamburger",
	"096": "hammock",
	"097": "harmonica",
	"098": "harp",
	"099": "harpsichord",
	"100": "hawksbill-101",
	"101": "head-phones",
	"102": "helicopter-101",
	"103": "hibiscus",
	"104": "homer-simpson",
	"105": "horse",
	"106": "horseshoe-crab",
	"107": "hot-air-balloon",
	"108": "hot-dog",
	"109": "hot-tub",
	"110": "hourglass",
	"111": "hosue-fly",
	"112": "human-skeleton",
	"113": "hummingbird",
	"114": "ibis-101",
	"115": "ice-cream-cone",
	"116": "iguana",
	"117": "ipod",
	"118": "iris",
	"119": "jesus-christ",
	"120": "joy-stick",
	"121": "kangaroo-101",
	"122": "kayak",
	"123": "ketch-101",
	"124": "killer-whale",
	"125": "knife",
	"126": "ladder",
	"127": "laptop-101",
	"128": "lathe",
	"129": "leopards-101",
	"130": "license-plate",
	"131": "lightbulb",
	"132": "light-house",
	"133": "lightning",
	"134": "llama-101",
	"135": "mailbox",
	"136": "mandolin",
	"137": "mars",
	"138": "mattress",
	"139": "megaphone",
	"140": "menorah-101",
	"141": "microscope",
	"142": "microwave",
	"143": "minaret",
	"144": "minotaur",
	"145": "motorbikes-101",
	"146": "mountain-bike",
	"147": "mushroom",
	"148": "mussels",
	"149": "necktie",
	"150": "octopus",
	"151": "ostrich",
	"152": "owl",
	"153": "palm-pilot",
	"154": "palm-tree",
	"155": "paperclip",
	"156": "paper-shredder",
	"157": "pci-card",
	"158": "penguin",
	"159": "people",
	"160": "pez-dispenser",
	"161": "photocopier",
	"162": "picnic-table",
	"163": "playing-card",
	"164": "porcupine",
	"165": "pram",
	"166": "praying-mantis",
	"167": "pyramid",
	"168": "raccoon",
	"169": "radio-telescope",
	"170": "rainbow",
	"171": "refrigerator",
	"172": "revolver-101",
	"173": "rifle",
	"174": "rotary-phone",
	"175": "roulette-wheel",
	"176": "saddle",
	"177": "saturn",
	"178": "school-bus",
	"179": "scorpion-101",
	"180": "screwdriver",
	"181": "segway",
	"182": "self-propelled-lawn-mower",
	"183": "sextant",
	"184": "sheet-music",
	"185": "skateboard",
	"186": "skunk",
	"187": "smokestack",
	"188": "snail",
	"189": "snail",
	"190": "snake",
	"191": "sneaker",
	"192": "snowmobile",
	"193": "soccer-ball",
	"194": "socks",
	"195": "soda-can",
	"196": "spaghetti",
	"197": "speed-boat",
	"198": "spider",
	"199": "spoon",
	"200": "stained-glass",
	"201": "starfish-101",
	"202": "steering-wheel",
	"203": "stirrups",
	"204": "sunflower-101",
	"205": "superman",
	"206": "sushi",
	"207": "swan",
	"208": "swiss-army-knife",
	"209": "sword",
	"210": "syringe",
	"211": "tambourine",
	"212": "teapot",
	"213": "teddy-bear",
	"214": "teepee",
	"215": "telephone-box",
	"216": "tennis-ball",
	"217": "tennis-court",
	"218": "tennis-racket",
	"219": "theodolite",
	"220": "toaster",
	"221": "tomato",
	"222": "tombstone",
	"223": "top-hat",
	"224": "touring-bike",
	"225": "tower-pisa",
	"226": "traffic-light",
	"227": "treadmill",
	"228": "triceratops",
	"229": "tricycle",
	"230": "trilobite-101",
	"231": "tripod",
	"232": "t-shirt",
	"233": "tuning-fork",
	"234": "tweezer",
	"235": "umbrella-101",
	"236": "unicorn",
	"237": "vcr",
	"238": "video-projector",
	"239": "washing-machine",
	"240": "watch-101",
	"241": "Waterfall",
	"242": "watermelon",
	"243": "welding-mask",
	"244": "wheelbarrow",
	"245": "windmill",
	"246": "wine-bottle",
	"247": "xylophone",
	"248": "yarmulke",
	"249": "yo-yo",
	"250": "zebra",
	"251": "airplanes-101",
	"252": "car-side-101",
	"253": "faces-easy-101",
	"254": "greyhound",
	"255": "tennis-shoes",
	"257": "clutter"}

def extract_color_stats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]
	# return our set of features
	return features



def load_images(images, labels):
	i = -1
	for imageFile in images:
		i = i + 1
		image = Image.open(sys.path[0] + "\\caltech 256\\256_ObjectCategories\\" + labels[i] + "." + objectDict[labels[i]] + "\\" + imageFile)
		features = extract_color_stats(image)
		label = labels[i]
		features.append(int(label))
		data.append(features)
	
def separateByClass(dataset): 
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def smpStdDev(values):
	avg = np.average(values)
	variance = sum([pow(x - avg, 2) for x in values])/float(len(values) - 1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(np.average(attribute), smpStdDev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProb(x, mean, stdDev):
	exp = math.exp(-(math.pow(x - mean, 2)/(2 * math.pow(stdDev, 2))))
	return (1/(math.sqrt(2 * math.pi)*stdDev))*exp

def calculateClassProbs(summaries, inputVector):
	probs = {}
	for classValue, classSummaries in summaries.items():
		probs[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdDev = classSummaries[i]
			x = inputVector[i]
			probs[classValue] *= calculateProb(x, mean, stdDev)	
	return probs

def predict(summaries, inputVector):
	probs = calculateClassProbs(summaries, inputVector)
	bestLabel = None
	bestProb = -1
	for classValue, prob in probs.items():
		if bestLabel is None or prob > bestProb:
			bestProb = prob
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	actualClasses = []
	correct = 0
	for x in range(len(testSet)):
		actualClasses.append(testSet[x][-1]);
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100.0, actualClasses

def splitDatasetStratified(dataset, splitRatio, classSizeList, classList):
	trainSize = classSizeList
	trainSet = []
	copy = list(dataset)
	for classIndex in range(len(classList)):
		trainSetTemp = []
		while len(trainSetTemp) < int(trainSize[classIndex] * splitRatio):
			index = random.randrange(len(copy))
			if copy[index][-1] == classList[classIndex]:
				temp = copy.pop(index)
				trainSetTemp.append(temp)
				trainSet.append(temp)
	return [trainSet, copy]

# here, copy is now trainSet
# for num in reversed(range())
def splitDatasetStratified_KFold(dataset, splitRatio, classSizeList, classList, round):
	trainSize = classSizeList
	testSet = []
	index = len(dataset) - 1 
	copy = list(dataset)
	interval = int(1/(1 - splitRatio))
	for classSize in reversed(classList):
		testSetTemp = []
		for counter in range(classSize * (1 - splitRatio)):
			temp = copy.pop(index)
			testSetTemp.append(temp)
			testSet.append(temp)
			index = index - interval
		index = len(dataset) - 1 - classSize
	return [copy, testSet]

def add_iso_class_to_testSet(testSet, classValue, newTestSet):
	for x in range(len(testSet)):
		if testSet[x][6] == classValue:
			newTestSet.append(testSet[x])
	return newTestSet

def loadFiles(label, start, size):
	for i in range(start, start+size):
		if (i > 0 and i <= 9):
			files.append(label + "_000" + str(i) + ".jpg") 
		if (i >= 10 and i <= 99):
			files.append(label + "_00" + str(i) + ".jpg")
		if (i >= 100 and i <= 999):
			files.append(label + "_0" + str(i) + ".jpg")
		labels.append(label)

def addAllFromTo(fromList, toList):
	while fromList != []:
		toList.append(fromList.pop(0))
	return toList

def totalPredictionAccuracy(length):
	total = length
	correct = 0
	for x in range(len(totalPredictions)):
		for y in range(len(totalPredictions[0])):
			if totalPredictions[x][y] >= 0 :
				correct = correct + 1
	return (correct/float(total))*100.0

def main():
	classList = ["069", "053", "113", "202", "096", "145"]
	# classList = ["069", "113", "202"]
	sizePerClass = 80
	splitRatio = .8
	numClassList = []
	sizeList = []
	for x in classList:
		loadFiles(x, 1, sizePerClass)
		numClassList.append(int(x))
		sizeList.append(sizePerClass)
	
	load_images(files, labels)
	trainingSet, testSet = splitDatasetStratified(data, splitRatio, sizeList, numClassList)
	print('Split {0} rows into trainSet = {1} and testSet = {2} rows'.format(len(data), len(trainingSet), len(testSet)))

	testSetSize = (sizePerClass - int(sizePerClass * splitRatio))
	for a in range(len(classList)):
		newList = [0] * testSetSize
		totalPredictions.append(newList)
		objToIndexDict[int(classList[a])] = a

	print(objToIndexDict)
	summaries = summarizeByClass(trainingSet) 
	# print(summaries)
	# predictions = getPredictions(summaries, testSet)
	# print(predictions)
	# accuracy = getAccuracy(testSet, predictions)
	# print("Accuracy: {0}%".format(accuracy))

##########
	for c1 in range(len(numClassList)):
		class1 = numClassList[c1]
		for class2 in numClassList[c1+1:]:
			tempSumarries = {}
			h = summaries[class1]
			tempSumarries[class1] = h
			h2 = summaries[class2]
			tempSumarries[class2] = h2
	
			newTestSet = []
			newTestSet = add_iso_class_to_testSet(testSet, class1, newTestSet)
			newTestSet = add_iso_class_to_testSet(testSet, class2, newTestSet)

			predictions = getPredictions(tempSumarries, newTestSet)
	
			accuracy, actualClasses = getAccuracy(newTestSet, predictions)
			print(str(class1) + ", " + str(class2))
			print("Accuracy: {0}%".format(accuracy))

			for i in range(len(predictions)):
				tpIndy = i % testSetSize
				tpIndx = objToIndexDict[actualClasses[i]]
				if predictions[i] == actualClasses[i]:
					totalPredictions[tpIndx][tpIndy] = totalPredictions[tpIndx][tpIndy] + 1
				else:
					totalPredictions[tpIndx][tpIndy] = totalPredictions[tpIndx][tpIndy] - 1
	
	print("Overall Accuracy: {}%".format(totalPredictionAccuracy(len(testSet))))
	
			

########################	

main()
