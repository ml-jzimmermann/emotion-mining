import pandas as pd


results = pd.read_csv("amt_results.csv").values
sentences = results[:, 27]
scores = results[:, 28]

observations = len(scores)

emotions = {
    "anticipation" : 0,
    "anger" : 0,
    "disgust" : 0,
    "sadness" : 0,
    "surprise" : 0,
    "fear" : 0,
    "trust" : 0,
    "joy" : 0,
    "neutral" : 0,
}

for score in scores:
    if score == 1:
        emotions["anticipation"] = emotions["anticipation"] + 1
    if score == 2:
        emotions["anger"] = emotions["anger"] + 1
    if score == 3:
        emotions["disgust"] = emotions["disgust"] + 1
    if score == 4:
        emotions["sadness"] = emotions["sadness"] + 1
    if score == 5:
        emotions["surprise"] = emotions["surprise"] + 1
    if score == 6:
        emotions["fear"] = emotions["fear"] + 1
    if score == 7:
        emotions["trust"] = emotions["trust"] + 1
    if score == 8:
        emotions["joy"] = emotions["joy"] + 1
    if score == 9:
        emotions["neutral"] = emotions["neutral"] + 1

print(emotions)

#for k in emotions.keys():
#    emotions[k] = emotions[k] / observations * 100

#print(emotions)

# plot
import matplotlib.pyplot as plt


title = "Proportion of Depression Sentences"
labels = ["anticipation", "anger", "disgust", "sadness", "surprise", "fear", "trust", "joy", "neutral"]
sizes = emotions.values()
colors = []
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0.05)

# plt.hist(emotions.values())

plot = plt.figure()
plt.axis("equal")
plt.title(title)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%', shadow=False, startangle=100)


# plt.bar(range(len(emotions)), list(emotions.values()), align="center")
# plt.xticks(range(len(emotions)), list(emotions.keys()))

# path = "../plots/" + title + ".png"
# plot.savefig(path)
plt.show()
