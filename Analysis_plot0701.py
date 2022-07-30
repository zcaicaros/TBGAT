import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

directory1 = "./results_PP/entropy_maac_pp5/"
#directory2 = "./results_pomm/DMAentropy_maac_pp5C_FFA_seeds/"
directory2 = "./results_PP/entropy_dmac_pp5/"
directory3 = "./results_PP/entropy_omac_pp5/"
directory4 = "./results_PP/entropy_domac_pp5/"
directory5 = "./results_pomm/DOMAC_FFA_QR3_seeds/"
MAAC_files = glob.glob(directory1 + 'train*.csv')
DMAC_files = glob.glob(directory2 + 'train*.csv')
OMAC_files = glob.glob(directory3 + 'train*.csv')
DOMAC_files = glob.glob(directory4 + 'train*.csv')
DOMACUB_files =  glob.glob(directory5 + 'train*.csv')
#file_results_MAAC
pd_maac = pd.DataFrame(columns=["Episode", "Score", "Seed"])
for maac_file in MAAC_files:
    pd_maac_file = pd.read_csv(maac_file, delimiter=";", names=["Episode", "Score"])
    pd_maac_file["Seed"] = maac_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_maac = pd.concat([pd_maac, pd_maac_file])
    pd_maac.reset_index()
pd_maac = pd_maac.reset_index(drop=True)
pd_maac_gb = pd_maac.groupby("Episode")
pd_maac_mean = pd_maac_gb.mean()
pd_maac_mean = pd_maac_mean.reset_index()
pd_maac_mean = pd_maac_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_maac_var = pd_maac_gb.var()
pd_maac_var = pd_maac_var.reset_index()
pd_maac_var = pd_maac_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})
x_maac = pd_maac_mean["Episode"].values
mean_maac = pd_maac_mean["MeanScore"].values
std_maac = np.sqrt(pd_maac_var["VarScore"].values)

#file_results_DMAC
pd_dmac = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for dmac_file in DMAC_files:
    pd_dmac_file = pd.read_csv(dmac_file, delimiter=";", names=["Episode", "Score"])
    pd_dmac_file["Seed"] = dmac_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_dmac = pd.concat([pd_dmac, pd_dmac_file])
    pd_dmac.reset_index()
pd_dmac = pd_dmac.reset_index(drop=True)
pd_dmac_gb = pd_dmac.groupby("Episode")
pd_dmac_mean = pd_dmac_gb.mean()
pd_dmac_mean = pd_dmac_mean.reset_index()
pd_dmac_mean = pd_dmac_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_dmac_var = pd_dmac_gb.var()
pd_dmac_var = pd_dmac_var.reset_index()
pd_dmac_var = pd_dmac_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})
x_dmac = pd_dmac_mean["Episode"].values
mean_dmac = pd_dmac_mean["MeanScore"].values
std_dmac = np.sqrt(pd_dmac_var["VarScore"].values)




#file_results OMAC
pd_omac = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for omac_file in OMAC_files:
    pd_omac_file = pd.read_csv(omac_file, delimiter=";", names=["Episode", "Score"])
    pd_omac_file["Seed"] = omac_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_omac = pd.concat([pd_omac, pd_omac_file])
    pd_omac.reset_index()

pd_omac = pd_omac.reset_index(drop=True)

pd_omac_gb = pd_omac.groupby("Episode")
#print('pd_train_gb',pd_train_gb)
pd_omac_mean = pd_omac_gb.mean()
#print('pd_train_mean',pd_train_mean)
pd_omac_mean = pd_omac_mean.reset_index()
pd_omac_mean = pd_omac_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_omac_var = pd_omac_gb.var()
pd_omac_var = pd_omac_var.reset_index()
pd_omac_var = pd_omac_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})
x_omac = pd_omac_mean["Episode"].values
mean_omac= pd_omac_mean["MeanScore"].values
std_omac= np.sqrt(pd_omac_var["VarScore"].values)

#file_results_DOMAC
pd_domac = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for domac_file in DOMAC_files:
    pd_domac_file = pd.read_csv(domac_file, delimiter=";", names=["Episode", "Score"])
    pd_domac_file["Seed"] = domac_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_domac = pd.concat([pd_domac, pd_domac_file])
    pd_domac.reset_index()

pd_domac = pd_domac.reset_index(drop=True)
pd_domac_gb = pd_domac.groupby("Episode")
#print('pd_test_gb',pd_test_gb)
pd_domac_mean = pd_domac_gb.mean()
pd_domac_mean = pd_domac_mean.reset_index()
pd_domac_mean = pd_domac_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_domac_var = pd_domac_gb.var()
pd_domac_var = pd_domac_var.reset_index()
pd_domac_var = pd_domac_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_domac = pd_domac_mean["Episode"].values
mean_domac = pd_domac_mean["MeanScore"].values
std_domac = np.sqrt(pd_domac_var["VarScore"].values)

#file_results_DOMAC_UB
pd_domacub = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for domacub_file in DOMACUB_files:
    pd_domacub_file = pd.read_csv(domacub_file, delimiter=";", names=["Episode", "Score"])
    pd_domacub_file["Seed"] = domacub_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_domacub = pd.concat([pd_domacub, pd_domacub_file])
    pd_domacub.reset_index()

pd_domacub = pd_domacub.reset_index(drop=True)
pd_domacub_gb = pd_domacub.groupby("Episode")
#print('pd_test_gb',pd_test_gb)
pd_domacub_mean = pd_domacub_gb.mean()
pd_domacub_mean = pd_domacub_mean.reset_index()
pd_domacub_mean = pd_domacub_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_domacub_var = pd_domacub_gb.var()
pd_domacub_var = pd_domacub_var.reset_index()
pd_domacub_var = pd_domacub_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_domacub = pd_domacub_mean["Episode"].values
mean_domacub = pd_domacub_mean["MeanScore"].values
std_domacub = np.sqrt(pd_domacub_var["VarScore"].values)



'''def custom_plot(x, y, z, xlabel, ylabel,title,color, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    #ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y, color)
    ax.fill_between(x, y - z/2, y + z/2, facecolor=base_line.get_color(), alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

custom_plot(x_test, mean_test, std_test, "# Episode", "Score", "Test score averaged over seeds".format(len(test_files)),"b",(10,5))
custom_plot(x_train, mean_train, std_train, "# Episode", "Score", "Train score averaged over seeds".format(len(train_files)), "r", (10,5))
'''
scale=4
import matplotlib.ticker as ticker
fig, ax = plt.subplots(figsize=(scale*1.618,scale))
x_lable_pomm = ['{}K'.format(int(x/1000)) for x in x_maac]

base_line1, = ax.plot(x_maac, mean_maac, 'gray', label= "MAAC")
base_line2, = ax.plot(x_dmac, mean_dmac, 'steelblue',label= 'DMAC')
base_line3, = ax.plot(x_omac, mean_omac, "olive",label= "OMAC")
base_line4, = ax.plot(x_domac, mean_domac, "lightcoral",label= "DOMAC")
#base_line5, = ax.plot(x_domacub, mean_domacub, "gray",label= "DOMAC(k=3)")
#base_line6, = ax.plot(x_madacopp2, mean_madacopp2, "seagreen",label= "maac_QRopp")
ax.fill_between(x_maac, mean_maac - std_maac/2, mean_maac + std_maac/2, facecolor=base_line1.get_color(), alpha=0.2)
ax.fill_between(x_dmac, mean_dmac - std_dmac/2, mean_dmac + std_dmac/2, facecolor=base_line2.get_color(), alpha=0.2)
ax.fill_between(x_omac, mean_omac- std_omac/2, mean_omac + std_omac/2, facecolor=base_line3.get_color(), alpha=0.2)
ax.fill_between(x_domac, mean_domac- std_domac/2, mean_domac + std_domac/2, facecolor=base_line4.get_color(), alpha=0.2)
#ax.fill_between(x_domacub, mean_domacub- std_domacub/2, mean_domacub + std_domacub/2, facecolor=base_line5.get_color(), alpha=0.2)
#ax.fill_between(x_madacopp2, mean_madacopp2- std_madacopp2/2, mean_madacopp2 + std_madacopp2/2, facecolor=base_line6.get_color(), alpha=0.5)
#plt.title("Average Return for PommeFFACompetitionFast-v0".format(len(MAAC_files)))
#plt.title("Accuarcy of estimated opponent model over 7*7".format(len(train_files)))

ax.legend(fontsize=12, loc = 'upper right')
ax.xaxis.set_major_formatter(ticker.EngFormatter())
#ax.set(ylabel='Average Return')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#ax.set(xlabel='Episode')
#axs.get_xaxis().set_visible(False)
ax.yaxis.label.set_size(15)
ax.grid(True)
#ax.set_xticklabels([])
ax.xaxis.label.set_size(15)


#plt.ylabel("Average reward")
#plt.ylabel("Cross entropy loss")
#plt.savefig('Averaged Reward-PommTeam_seed.png')

plt.tight_layout()
#plt.savefig('Averaged Reward-PP4v2_seed.pdf')
#plt.savefig('Averaged Reward-PommFFA_seed.pdf')
#plt.savefig('Trained OM analysis in Pomm-FFA_new1.pdf')
plt.savefig('entropy of agent in PP2v1.pdf')
plt.show()