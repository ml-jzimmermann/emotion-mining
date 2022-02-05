import pandas as pd


csv = pd.read_csv("primary-plutchik-wheel-DFE.csv")
sentences = csv["sentence"]
results = csv["emotion"]
confidence = csv["emotion:confidence"]

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

for i in results:
    if i == "Anticipation":
        emotions["anticipation"] = emotions["anticipation"] + 1
    if i == "Anger":
        emotions["anger"] = emotions["anger"] + 1
    if i == "Disgust":
        emotions["disgust"] = emotions["disgust"] + 1
    if i == "Sadness":
        emotions["sadness"] = emotions["sadness"] + 1
    if i == "Surprise":
        emotions["surprise"] = emotions["surprise"] + 1
    if i == "Fear":
        emotions["fear"] = emotions["fear"] + 1
    if i == "Trust":
        emotions["trust"] = emotions["trust"] + 1
    if i == "Joy":
        emotions["joy"] = emotions["joy"] + 1
    if i == "Neutral":
        emotions["neutral"] = emotions["neutral"] + 1

print(emotions)
import matplotlib.pyplot as plt


title = "Proportion of Example-Dataset Emotions"
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
