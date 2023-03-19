import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

r = 100

def make_df(data_size):
        
    X1 = []
    X2 = []
    X3 = []
    Y = []



    for i in range(data_size):
        x1 = np.random.randint(-(r),r)
        x2 = np.random.randint(-(r),r)
        x3 = np.random.randint(-(r),r)

        # y = (x1)+ (x2)
        y = (x1)**3 + (x2)**2 + (x3+6)
        # y = x1**2 + x2 


        X1.append(x1)
        X2.append(x2)
        X3.append(x3)

        Y.append(y)

    # df = pd.DataFrame({"X1":X1, "X2":X2, "y":Y})
    df = pd.DataFrame({"X1":X1, "X2":X2, "X3":X3, "y":Y})

    return df

df = make_df(10000)
df.to_csv("hard_polynomial_train_data.csv")

df = make_df(1000)
df.to_csv("hard_polynomial_test_data.csv")

# y = df["y"]
# x = np.arange(df.shape[0])
# sns.lineplot(x = x,y = y)
# plt.show()

# x = np.array(df.drop("y", axis = 1))
# y = np.array(df["y"])

p = np.array([[i,i,i] for i in range(-r,r)])
y = np.array([[(i[0]**3) + i[1]**2 + i[2] + 6] for i in p])
plt.plot(range(-r,r),y,linewidth = '10')
plt.plot(range(-r,r),y)
plt.show()