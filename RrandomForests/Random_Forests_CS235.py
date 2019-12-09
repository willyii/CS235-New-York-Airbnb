import pandas as pd
import numpy as np
from random import randrange


def OneHotEncoding(x):
	return np.array(pd.get_dummies(x))


def Max_min_norm(x):
	return np.array((x.values - x.min()) / (x.max() - x.min())).reshape(-1, 1)


def Log_norm(x):
	return np.array(np.log(x + 1) / np.log(x.max())).reshape(-1, 1)


def Z_Score(x):
	return np.array((x.values - x.mean()) / x.std()).reshape(-1, 1)


def Fillna_with_Min(x):
	return x.fillna(x.min())


def normalize(matrix):
	norm = (np.linalg.norm(matrix, axis=0))
	norm[norm == 0] = 1
	normalized_matrix = matrix / norm
	return normalized_matrix, norm



def load_data(path):
	# load data
	csv = pd.read_csv(path)
	# prepare data
	group = OneHotEncoding(csv['neighbourhood_group'])
	room_type = OneHotEncoding(csv['room_type'])
	latitude = Max_min_norm(csv['latitude'])
	longitude = Max_min_norm(csv['longitude'])
	host_listings_count = Max_min_norm(csv['calculated_host_listings_count'])
	number_of_reviews = Max_min_norm(csv['number_of_reviews'])
	availability = Max_min_norm(csv['availability_365'])
	nights = Log_norm(csv['minimum_nights'])
	last_view = Log_norm(csv['last_review'])

	# reviews_per_month = Log_norm(csv['reviews_per_month'])
	csv['reviews_per_month'] = Fillna_with_Min(csv['reviews_per_month'])
	reviews_per_month = Log_norm(csv['reviews_per_month'])
	price = np.array(csv['price']).reshape(-1, 1)
	price = np.log1p(price)

	data = np.hstack((
		group, room_type, latitude, longitude, nights, number_of_reviews,
		host_listings_count, availability,last_view, reviews_per_month, price
	))
	dataset = pd.DataFrame(data)
	# dataset = np.random.rand(50, 60)
	# label = np.sum(dataset, axis=1).reshape((50, 1))
	# dataset_df = pd.DataFrame(dataset)
	# dataset_df["label"] = label

	return dataset

class RandomForests:
	def __init__(self, n_folds, max_depth, min_size, sample_size, n_trees, dataset):
		self.n_folds = n_folds
		self.max_depth = max_depth
		self.min_size = min_size
		self.sample_size = sample_size
		self.n_trees = n_trees
		self.dataset = dataset
		self.n_features = int(dataset.shape[1] - 1)

	# Split a dataset into k folds
	def cross_validation_split(self, dataset, n_folds):
		dataset_split = list()
		dataset_copy = list(dataset)
		fold_size = int(len(dataset) / n_folds)
		for i in range(n_folds):
			fold = list()
			while len(fold) < fold_size:
				index = randrange(len(dataset_copy))
				fold.append(dataset_copy.pop(index))
			dataset_split.append(fold)
		return dataset_split

	# Calculate accuracy percentage
	def accuracy_metric(self, actual, predicted):
		return np.mean((actual - predicted) ** 2)

	# Evaluate an algorithm using a cross validation split
	def evaluate_algorithm(self):
		folds = self.cross_validation_split(self.dataset, self.n_folds)
		scores = list()
		for i in range(len(folds)):
			train_set = list(folds)
			train_set.pop(i)
			train_set = sum(train_set, [])
			test_set = list()
			for row in folds[i]:
				row_copy = list(row)
				test_set.append(row_copy)
				row_copy[-1] = None
			predicted = self.random_forest(train_set, test_set, self.max_depth, self.min_size, self.sample_size, self.n_trees, self.n_features)
			actual = [row[-1] for row in folds[i]]
			accuracy = self.accuracy_metric(np.asarray(actual), predicted)
			scores.append(accuracy)
		return scores

	# Split a dataset based on an attribute and an attribute value
	def test_split(self, index, value, dataset):
		left, right = list(), list()
		for row in dataset:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right

	# Calculate the Gini index for a split dataset
	def gini_index(self, groups, classes):

		if groups[0] != []:
			left = np.asarray(groups[0])[:, -1]
			left_mean = np.mean(left)
			left_value = np.mean((left - left_mean) ** 2)
		else:
			left_mean = 0
			left_value = 0

		if groups[1] != []:
			right = np.asarray(groups[1])[:, -1]
			right_mean = np.mean(right)
			right_value = np.mean((right - right_mean) ** 2)
		else:

			right_mean = 0
			right_value = 0

		total = np.asarray(groups[0] + groups[1])[:, -1]
		total_mean = np.mean(total)
		total_value = np.mean((total - total_mean) ** 2)

		return left_value + right_value - total_value

	# Select the best split point for a dataset
	def get_split(self, dataset, n_features):
		class_values = list(set(row[-1] for row in dataset))
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		features = list()
		while len(features) < n_features:
			index = randrange(len(dataset[0]) - 1)
			if index not in features:
				features.append(index)

		for feature_index in features:
			split_candidate = np.unique(np.asarray(dataset)[:, feature_index])
			if len(split_candidate) > 100:
				split_candidate = np.random.choice(split_candidate, 100)
			for split_value in split_candidate:
				groups = self.test_split(feature_index, split_value, dataset)
				gini = self.gini_index(groups, class_values)
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, split_value, gini, groups
		return {'index': b_index, 'value': b_value, 'groups': b_groups}

	# Create a terminal node value
	def to_terminal(self, group):
		# outcomes = [row[-1] for row in group]
		return np.mean(np.asarray(group)[:, -1])

	# Create child splits for a node or make terminal
	def split(self, node, max_depth, min_size, n_features, depth):
		if node["groups"] == None:
			return
		left, right = node['groups']
		del (node['groups'])
		# check for a no split
		if not left or not right:
			node['left'] = node['right'] = self.to_terminal(left + right)
			return
		# check for max depth
		if depth >= max_depth:
			node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
			return
		# process left child
		if len(left) <= min_size:
			node['left'] = self.to_terminal(left)
		else:
			node['left'] = self.get_split(left, n_features)
			self.split(node['left'], max_depth, min_size, n_features, depth + 1)
		# process right child
		if len(right) <= min_size:
			node['right'] = self.to_terminal(right)
		else:
			node['right'] = self.get_split(right, n_features)
			self.split(node['right'], max_depth, min_size, n_features, depth + 1)

	# Build a decision tree
	def build_tree(self, train, max_depth, min_size, n_features):
		root = self.get_split(train, n_features)
		self.split(root, max_depth, min_size, n_features, 1)
		return root

	# Make a prediction with a decision tree
	def predict(self, node, row):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict):
				return self.predict(node['left'], row)
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return self.predict(node['right'], row)
			else:
				return node['right']

	# Create a random subsample from the dataset with replacement
	def subsample(self, dataset, ratio):
		sample = list()
		n_sample = round(len(dataset) * ratio)
		while len(sample) < n_sample:
			index = randrange(len(dataset))
			sample.append(dataset[index])
		return sample

	# Make a prediction with a list of bagged trees
	def bagging_predict(self, trees, row):
		predictions = [self.predict(tree, row) for tree in trees]
		return np.mean(predictions)

	# Random Forest Algorithm
	def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
		trees = list()
		for i in range(n_trees):
			sample = self.subsample(train, sample_size)
			tree = self.build_tree(sample, max_depth, min_size, n_features)
			trees.append(tree)
		predictions = [self.bagging_predict(trees, row) for row in test]
		return (predictions)


if __name__ == '__main__':
	path = 'CleanedData.csv'
	dataset = load_data(path)
	n_folds = 5
	max_depth = 5
	min_size = 1
	sample_size = 0.7
	n_trees = 5
	method = RandomForests(n_folds, max_depth, min_size, sample_size,  n_trees, dataset.to_numpy())
	scores = method.random_forest()
	print('Mean Squared error: %.3f%%' % np.mean(scores))

