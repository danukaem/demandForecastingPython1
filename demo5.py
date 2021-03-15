import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer


porter = PorterStemmer()
lancaster=LancasterStemmer()
print("Porter Stemmer")
print(porter.stem("cats"))
print(porter.stem("trouble"))
print(porter.stem("troubling"))
print(porter.stem("troubled"))
print("Lancaster Stemmer")
print(lancaster.stem("cats"))
print(lancaster.stem("trouble"))
print(lancaster.stem("troubling"))
print(lancaster.stem("troubled"))

print(porter.stem("calculating"))
print(porter.stem("calculated"))

print(lancaster.stem("calculating"))
print(lancaster.stem("calculated"))


#
# lemmatizer = WordNetLemmatizer()
# print("rocks :", lemmatizer.lemmatize("playing",'v'))
# print("rocks :", lemmatizer.lemmatize("are",'v'))
# print("rocks :", lemmatizer.lemmatize("car cars car's",'n'))


# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-5, 5, 0.15)
# Y = np.arange(-5, 5, 0.15)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
#
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
# plt.show()

# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("cacti"))
# print(lemmatizer.lemmatize("geese"))
# print(lemmatizer.lemmatize("rocks"))
# print(lemmatizer.lemmatize("python"))
# print(lemmatizer.lemmatize("better", pos="a"))
# print(lemmatizer.lemmatize("best", pos="a"))
# print(lemmatizer.lemmatize("run"))
# print(lemmatizer.lemmatize("running",'v'))
# print(lemmatizer.lemmatize("ran",'v'))