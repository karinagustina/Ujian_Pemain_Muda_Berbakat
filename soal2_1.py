import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
# print(df)

tAge = df[['Age']][df['Age'] <= 25]
print(len(tAge))
tOverall = df[['Overall']][df['Overall'] >= 80]   
print(len(tOverall))
tPotential = df[['Potential']][df['Potential'] >= 80]

ntAge = df[['Age']][df['Age'] > 25]  
ntOverall = df[['Overall']][df['Overall'] < 80]    
ntPotential = df[['Potential']][df['Potential'] < 80]   

fig = plt.figure('Fifa', figsize = (13,6))

ax = plt.subplot(121)
# plt.scatter(
#     tAge,
#     tOverall,
#     color = 'g',
#     label = 'Target'
# )
# plt.scatter(
#     ntAge,
#     ntOverall,
#     color = 'r',
#     label = 'Non-Target'
# )
ax.legend()
ax.grid(True)
ax.set_xlabel('Age')
ax.set_ylabel('Overall')
ax.set_title('Age vs Overall')
ax.legend()
ax.grid(True)

bx = plt.subplot(122)
plt.scatter(
    tAge,
    tPotential,
    color = 'g',
    label = 'Target'
)
plt.scatter(
    ntAge,
    ntPotential,
    color = 'r',
    label = 'Non-Target'
)
bx.set_xlabel('Age')
bx.set_ylabel('Potential')
bx.set_title('Age vs Potential')
bx.legend()
bx.grid(True)

plt.show()