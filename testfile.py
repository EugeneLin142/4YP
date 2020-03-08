from eval_functions import eval_model

eval_model(model_name="doc2vec", dimres_method="pca", model_epochs=100, model_dim_size=300, db_params=[0.2,1], drawplots=0)

# import numpy as np; np.random.seed(0)
# import seaborn as sns; sns.set()
# import pandas as pd
# import matplotlib.pyplot as plt
#
# data = [[1,2], [2,3], [3,4]]
#
# data = pd.DataFrame(data, columns=["col1", "col2"], index=["r1", "r2", "r3"])
#
# print(data)
# ax = sns.heatmap(data, annot=True)
# # fix for mpl bug that cuts off top/bottom of seaborn viz
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t) # update the ylim(bottom, top) values
# plt.show()
# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame(data=None)
# df["column one"] = [[1, 4], [2, 3], [3, 4]]
#
# print(df["column one"].tolist().index([1, 4]))
#



# print(common_texts[0])
# print(len(common_texts))
#
# model = FastText(size=5, window=3, min_count=1)
# model.build_vocab(sentences=common_texts)
# model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)
#
# print(model.wv["human"])
# print(model.wv.vectors_vocab[0])


# exist_word = "computer"
# computer_vec = model.wv[exist_word]
# print(computer_vec)
#
# oov_word = "graph-out-of-vocab"
#
# oov_vec = model.wv[oov_word]
# print(oov_vec)