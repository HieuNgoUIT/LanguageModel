from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

# Create a placeholder for model
model = defaultdict(lambda: defaultdict(lambda: 0))


# print(trigrams)

# Count frequency of co-occurance  
for sentence in reuters.sents():
        # print("________________")
        # print(sentence)
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1

#print(model[('diluted', '41')]['cts'])
# for k,v in model.items():
#         print('key: {} value: {} '.format(k,v))

# Let's transform the counts to probabilities

for w1_w2 in model:
	#print("w1_w2: ",w1_w2)
	#print("model[w1_w2].values: ",model[w1_w2].values())
	total_count = float(sum(model[w1_w2].values()))
	for w3 in model[w1_w2]:
		model[w1_w2][w3] /= total_count
		#print("model[w1_w2][w3].values: ",model[w1_w2][w3])
print(dict(model["price", "for"]))


