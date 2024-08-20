################################################## Path
# Path
download = 


################################################## Téléchargement des packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from umap import UMAP
from functools import reduce
from mlxtend.plotting import plot_pca_correlation_graph
from PIL import Image
from IPython.display import Image, display



################################################## Téléchargement des données
revcommunes = pd.read_csv(download+"revcommunes.csv")
agesexcommunes= pd.read_csv(download+"agesexcommunes.csv")
diplomescommunes= pd.read_csv(download+"diplomescommunes.csv")
proprietairescommunes= pd.read_csv(download+"proprietairescommunes.csv")
cspcommunes= pd.read_csv(download+"cspcommunes.csv")
capitalimmobiliercommunes= pd.read_csv(download+"capitalimmobiliercommunes.csv")
pres2022comm = pd.read_csv(download+"pres2022comm.csv")
pres2017comm = pd.read_csv(download+"pres2017comm.csv")
pres2012comm = pd.read_csv(download+"pres2012comm.csv")
pres2007comm = pd.read_csv(download+"pres2007comm.csv")
pres2002comm = pd.read_csv(download+"pres2002comm.csv")
pres1995comm = pd.read_csv(download+"pres1995comm.csv")
pres1988comm = pd.read_csv(download+"pres1988comm.csv")
pres1981comm = pd.read_csv(download+"pres1981comm.csv")
pres1974comm = pd.read_csv(download+"pres1974comm.csv")
pres1969comm = pd.read_csv(download+"pres1969comm.csv")
pres1965comm = pd.read_csv(download+"pres1965comm.csv")
pres1848comm = pd.read_csv(download+"pres1848comm.csv")



################################################## Création de données 2022

# Sélection des colonnes
revenu_2022 = revcommunes[['codecommune','revmoyadu2022', 'revmoyfoy2022']]
vote_2022 = pres2022comm[['dep','codecommune', 'inscrits', 'votants', 'exprimes', 'voteG', 
                          'voteCG', 'voteC', 'voteCD', 'voteD', 'pvoteG', 'pvoteCG', 'pvoteC', 
                          'pvoteCD', 'pvoteD']]
age_2022 = agesexcommunes[['codecommune', 'ageh2022', 'agef2022']]
diplome_2022 = diplomescommunes[['codecommune', 'pbac2022']]
proprietaire_2022 = proprietairescommunes[['codecommune', 'ppropri2022']]
csp_2022 = cspcommunes[['codecommune', 'pagri2022', 'pindp2022', 'pcadr2022', 
                        'ppint2022', 'pempl2022', 'pouvr2022']]
population_2022 = capitalimmobiliercommunes[['codecommune','pop2022']]

# Changement de datatype
vote_2022.loc[:,'codecommune'] = vote_2022['codecommune'].astype(str)
revenu_2022.loc[:,'codecommune'] = revenu_2022['codecommune'].astype(str)
age_2022.loc[:,'codecommune'] = age_2022['codecommune'].astype(str)
diplome_2022.loc[:,'codecommune'] = diplome_2022['codecommune'].astype(str)
proprietaire_2022.loc[:,'codecommune'] = proprietaire_2022['codecommune'].astype(str)
csp_2022.loc[:,'codecommune'] = csp_2022['codecommune'].astype(str)
population_2022.loc[:,'codecommune'] = population_2022['codecommune'].astype(str)

# Vote
vote_2022.loc[vote_2022['pvoteG'].isna(), # commnes avec vote exprmiées nul = 0
              ['pvoteG', 'pvoteCG', 'pvoteC', 'pvoteCD', 'pvoteD']] = 0
def get_vote_class(row): # Fonction pour choisir la vote majoritaire
    return max(['pvoteC', 'pvoteCG', 'pvoteC', 'pvoteCD', 'pvoteD'], key=lambda col: row[col])
vote_2022['vote_majoritaire'] = vote_2022.apply(get_vote_class, axis=1) # Création colonne vote majoritaire
vote_2022['votants/inscrits'] = vote_2022['votants']/vote_2022['inscrits']
vote_2022['participation'] = pd.qcut(vote_2022['votants/inscrits'], # Création de colonne quartile
                                     q=4, 
                                     labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Revenu 
revenu_2022.dropna(inplace = True) # Suppression des communes qui n'ont pas d'info de revenu
revenu_2022['revenu'] = pd.qcut(revenu_2022['revmoyadu2022'], # Création de colonne quartile
                                q=4, 
                                labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Diplome
diplome_2022.dropna(inplace = True) # Suppression des communes sans info diplome
diplome_2022['diplome'] = pd.qcut(diplome_2022['pbac2022'], # Création de colonne quartile
                                q=4, 
                                labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Propriétaire
proprietaire_2022.dropna(inplace = True) # Suppression des communes sans info proprietaire
proprietaire_2022['proprietaire'] = pd.qcut(proprietaire_2022['ppropri2022'], # Création de colonne quartile
                                q=4, 
                                labels=['Q1', 'Q2', 'Q3', 'Q4'])

# CSP
csp_2022.dropna(inplace = True) # Suppression des communes sans info CSP
def get_csp_class(row): # Fonction pour choisir la csp majoritaire
    return max(['pagri2022', 'pindp2022', 'pcadr2022', 'ppint2022', 'pempl2022', 'pouvr2022'], key=lambda col: row[col])
csp_2022['csp'] = csp_2022.apply(get_csp_class, axis=1) # Création colonne csp majoritaire

# Population
population_2022['population'] = pd.qcut(population_2022['pop2022'], # Création de colonne quartile
                                q=4, 
                                labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Combinaison de données : revenu + vote
df_2022_vote_revenu = pd.merge(vote_2022, revenu_2022, on='codecommune', how='outer')
df_2022_vote_revenu = df_2022_vote_revenu[~df_2022_vote_revenu[['dep','revmoyadu2022', 'revmoyfoy2022']].isna().any(axis = 1)]
df_2022_vote_revenu.dropna(inplace=True)

# Combinaison de données : revenu + vote + age + diplome + proprietaire + csp
dataframes = [vote_2022, revenu_2022, age_2022, diplome_2022, proprietaire_2022, csp_2022, population_2022]
df_2022 = reduce(lambda left, right: pd.merge(left, right, on='codecommune', how='outer'), dataframes)
df_2022.dropna(inplace=True)
df_2022.reset_index(drop=True, inplace=True)



################################################## Création de données des autres années
final_dfs = {}

for year, df in zip([1981, 1988, 1995, 2002, 2007, 2012, 2017],
                    [pres1981comm, pres1988comm, pres1995comm, pres2002comm, 
                     pres2007comm, pres2012comm, pres2017comm]):
    # Select relevant columns from each DataFrame
    vote = df[['dep', 'codecommune', 'inscrits', 'votants', 'exprimes', 'voteG', 'voteCG', 
               'voteC', 'voteCD', 'voteD', 'pvoteG', 'pvoteCG', 'pvoteC', 'pvoteCD', 'pvoteD']]
    revenu = revcommunes[['codecommune', 'revmoyadu' + str(year), 'revmoyfoy' + str(year)]]
    age = agesexcommunes[['codecommune', 'ageh' + str(year), 'agef' + str(year)]]
    diplome = diplomescommunes[['codecommune', 'pbac' + str(year)]]
    proprietaire = proprietairescommunes[['codecommune', 'ppropri' + str(year)]]
    csp = cspcommunes[['codecommune', 'pagri' + str(year), 'pindp' + str(year), 'pcadr' + str(year), 
                       'ppint' + str(year), 'pempl' + str(year), 'pouvr' + str(year)]]
    population = capitalimmobiliercommunes[['codecommune', 'pop' + str(year)]]
    # Change datatype of 'codecommune' to str
    for df in [vote, revenu, age, diplome, proprietaire, csp, population]:
        df['codecommune'] = df['codecommune'].astype(str) 
    # Process vote data
    vote.loc[vote['pvoteG'].isna(), ['pvoteG', 'pvoteCG', 'pvoteC', 'pvoteCD', 'pvoteD']] = 0
    vote['vote_majoritaire'] = vote.apply(lambda row: max(['pvoteG', 'pvoteCG', 'pvoteC', 'pvoteCD', 'pvoteD'], key=lambda col: row[col]), axis=1)
    vote['votants/inscrits'] = vote['votants'] / vote['inscrits']
    vote['participation'] = pd.qcut(vote['votants/inscrits'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    # Process revenu data
    revenu.dropna(inplace=True)
    revenu['revenu'] = pd.qcut(revenu['revmoyadu' + str(year)], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  
    # Process diplome data
    diplome.dropna(inplace=True)
    diplome['diplome'] = pd.qcut(diplome['pbac' + str(year)], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])   
    # Process proprietaire data
    proprietaire.dropna(inplace=True)
    proprietaire['proprietaire'] = pd.qcut(proprietaire['ppropri' + str(year)], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  
    # Process csp data
    csp.dropna(inplace=True)
    csp['csp'] = csp.apply(lambda row: max(['pagri' + str(year), 'pindp' + str(year), 'pcadr' + str(year), 
                                            'ppint' + str(year), 'pempl' + str(year), 'pouvr' + str(year)], key=lambda col: row[col]), axis=1)    
    # Process population data
    population['population'] = pd.qcut(population['pop' + str(year)], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  
    # Combine all data
    dataframes = [vote, revenu, age, diplome, proprietaire, csp, population]
    final_df = reduce(lambda left, right: pd.merge(left, right, on='codecommune', how='outer'), dataframes)
    final_df.dropna(inplace=True)
    final_df.reset_index(drop=True, inplace=True)
  
    # Store the final DataFrame in the dictionary with a dynamic name
    final_dfs[year] = final_df

df_1981 = final_dfs[1981]
df_1988 = final_dfs[1988]
df_1995 = final_dfs[1995]
df_2002 = final_dfs[2002]
df_2007 = final_dfs[2007]
df_2012 = final_dfs[2012]
df_2017 = final_dfs[2017]



################################################## Aperçu de données
print(df_2022.head(5).transpose().to_string(header=False))



################################################## Sélection de données vote & revenu
df_2022_vote_revenu = pd.merge(vote_2022, revenu_2022, on='codecommune', how='outer')
df_2022_vote_revenu = df_2022_vote_revenu[~df_2022_vote_revenu[['dep','revmoyadu2022', 'revmoyfoy2022']].isna().any(axis = 1)]
df_2022_vote_revenu.dropna(inplace=True)
df_2022_vote_revenu.head()
print(df_2022_vote_revenu.head().transpose().to_string(header=False))



################################################## Standardisation de données
scaler = StandardScaler()

df_2022_vote_revenu_norm_fit = scaler.fit_transform(df_2022_vote_revenu.drop(['dep', 'codecommune',
                                                                              'vote_majoritaire',
                                                                              'participation','revenu'], 
                                                                              axis = 1))
col_names_vote_revenu = ['inscrits', 'votants', 'exprimes', 'voteG','voteCG', 'voteC', 
                         'voteCD', 'voteD', 'pvoteG', 'pvoteCG', 'pvoteC', 'pvoteCD', 
                         'pvoteD', 'votants/inscrits', 'revmoyadu2022','revmoyfoy2022']
df_2022_vote_revenu_norm = pd.merge(df_2022[['dep', 'codecommune','vote_majoritaire','participation','revenu']], 
                                    pd.DataFrame(df_2022_vote_revenu_norm_fit, 
                                                 columns = col_names_vote_revenu),
                                    left_index=True, 
                                    right_index=True)

df_2022_norm_fit = scaler.fit_transform(df_2022.drop(['dep', 'codecommune','vote_majoritaire',
                                                      'participation','revenu', 'diplome', 
                                                      'proprietaire','csp', 'population'], 
                                                    axis = 1))
col_names = ['inscrits', 'votants', 'exprimes', 'voteG','voteCG', 'voteC', 'voteCD', 'voteD', 
             'pvoteG', 'pvoteCG', 'pvoteC', 'pvoteCD', 'pvoteD', 'votants/inscrits',
             'revmoyadu2022','revmoyfoy2022', 'ageh2022', 'agef2022', 'pbac2022','ppropri2022', 
             'pagri2022', 'pindp2022','pcadr2022', 'ppint2022', 'pempl2022', 
             'pouvr2022','pop2022']
df_2022_norm = pd.merge(df_2022[['dep', 'codecommune','vote_majoritaire','participation',
                                 'revenu', 'diplome', 'proprietaire','csp', 'population']], 
                        pd.DataFrame(df_2022_norm_fit, columns = col_names),
                        left_index=True, 
                        right_index=True)



################################################## PCA (vote & revenu 2022)
pca = PCA(n_components = 2, random_state = 0)

# PCA sur les variables vote et revenu seuls
pca_2022_vote_revenu = pca.fit_transform(df_2022_vote_revenu_norm.drop(['dep', 'codecommune',
                                                                        'vote_majoritaire',
                                                                        'participation',
                                                                        'revenu'], 
                                                                        axis = 1))
df_2022_vote_revenu_norm['x_pca'] = pca_2022_vote_revenu[:,0]  
df_2022_vote_revenu_norm['y_pca'] = pca_2022_vote_revenu[:,1]

hues_vote_revenu = ['vote_majoritaire', 'revenu']

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.suptitle("PCA")
axes = axes.flatten()
for i, hue in enumerate(hues_vote_revenu):
    ax = axes[i]
    sns.scatterplot(ax=ax, x='x_pca', y='y_pca', data=df_2022_vote_revenu_norm, hue=hue, 
                    palette='Set3', linewidth = 0.2)
    ax.set_title(f"{hue}", color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc = 'lower right', facecolor = 'white', edgecolor = 'none')
    ax.set_xlabel("Principal Component 1", linespacing = 0.5)
    ax.set_ylabel("Principal Component 2", linespacing = 0.5)
plt.show()



################################################## T-SNE (vote & revenu 2022)
tsne = TSNE(perplexity = 15, random_state = 0)

# t-SNE sur les variables vote et revenu seuls
tsne_2022_vote_revenu = tsne.fit_transform(df_2022_vote_revenu.drop(['dep', 'codecommune',
                                                                     'vote_majoritaire',
                                                                     'participation',
                                                                     'revenu'], axis = 1))
df_2022_vote_revenu['x_tsne'] = tsne_2022_vote_revenu[:,0]  
df_2022_vote_revenu['y_tsne'] = tsne_2022_vote_revenu[:,1]
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
fig.suptitle("t-SNE")
axes = axes.flatten()
for i, hue in enumerate(hues_vote_revenu):
    ax = axes[i]
    sns.scatterplot(ax=ax, x='x_tsne', y='y_tsne', data=df_2022_vote_revenu, hue=hue, 
                    palette='Set3', linewidth = 0.2)
    ax.set_title(f"{hue}", fontsize = 10, color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc = 'lower right', facecolor = 'white', edgecolor = 'none')
    ax.set_xlabel("t-SNE Component 1", linespacing = 0.5)
    ax.set_ylabel("t-SNE Component 2", linespacing = 0.5)
plt.show()



################################################## UMAP (vote & revenu 2022)
umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)

# UMAP sur les variables vote et revenu seuls
umap_2022_vote_revenu = umap.fit_transform(df_2022_vote_revenu_norm.drop(['dep', 'codecommune',
                                                                          'vote_majoritaire',
                                                                          'participation',
                                                                          'revenu', 'x_pca',
                                                                          'y_pca'], axis = 1))
df_2022_vote_revenu_norm['x_umap'] = umap_2022_vote_revenu[:,0]  
df_2022_vote_revenu_norm['y_umap'] = umap_2022_vote_revenu[:,1]

hues_vote_revenu = ['vote_majoritaire', 'revenu']

fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
fig.suptitle("UMAP")
axes = axes.flatten()
for i, hue in enumerate(hues_vote_revenu):
    ax = axes[i]
    sns.scatterplot(ax=ax, x='x_umap', y='y_umap', data=df_2022_vote_revenu_norm, hue=hue, 
                    palette='Set3', linewidth = 0.2)
    ax.set_title(f"{hue}", color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc = 'upper left', facecolor = 'white', edgecolor = 'none')
    ax.set_xlabel("UMAP Component 1", linespacing = 0.5)
    ax.set_ylabel("UMAP Component 2", linespacing = 0.5)
plt.show()



################################################## PCA (toutes les variables 2022)

# PCA sur toutes les variables
pca_2022 = pca.fit_transform(df_2022_norm.drop(['dep', 'codecommune','vote_majoritaire',
                                                'participation','revenu', 'diplome', 
                                                'proprietaire','csp', 'population'], 
                                                axis = 1))
df_2022_norm['x_pca'] = pca_2022[:,0]  
df_2022_norm['y_pca'] = pca_2022[:,1]

hues = ['vote_majoritaire', 'csp','revenu', 'diplome']

fig, axes = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('PCA', fontsize = 10)
axes = axes.flatten()
for i, hue in enumerate(hues):
    ax = axes[i]
    sns.scatterplot(ax=ax, x='x_pca', y='y_pca', data=df_2022_norm, hue=hue, palette='Set3', 
                    linewidth = 0.2)
    ax.set_title(f"{hue}", color='dimgray', size = 10)
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc = 'lower right', facecolor = 'white', edgecolor = 'none', fontsize = 7)
    ax.set_xlabel("Principal Component 1", linespacing = 0.5, size = 8)
    ax.set_ylabel("Principal Component 2", linespacing = 0.5, size = 8)
plt.show()



################################################## t-SNE (toutes les variables 2022)

tsne_2022 = tsne.fit_transform(df_2022.drop(['dep', 'codecommune','vote_majoritaire',
                                             'participation','revenu', 'diplome', 
                                             'proprietaire','csp', 'population'], 
                                             axis = 1))
df_2022['x_tsne'] = tsne_2022[:,0]  
df_2022['y_tsne'] = tsne_2022[:,1]
 
fig, axes = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('t-SNE', fontsize = 10)
axes = axes.flatten()
for i, hue in enumerate(hues):
    ax = axes[i]
    sns.scatterplot(ax=ax, x='x_tsne', y='y_tsne', data=df_2022, hue=hue, palette='Set3',
                    linewidth = 0.2)
    ax.set_title(f"{hue}", size = 10, color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc = 'lower right', facecolor = 'white', edgecolor = 'none', fontsize = 7)
    ax.set_xlabel("t-SNE Component 1", size = 8, linespacing = 0.5)
    ax.set_ylabel("t-SNE Component 2", size = 8, linespacing = 0.5)
plt.show()



################################################## UMAP (toutes les variables 2022)

umap_2022 = umap.fit_transform(df_2022_norm.drop(['dep', 'codecommune','vote_majoritaire',
                                                  'participation','revenu', 'diplome', 
                                                  'proprietaire','csp', 'population',
                                                  'x_pca', 'y_pca'], axis=1))
df_2022_norm['x_umap'] = umap_2022[:, 0]
df_2022_norm['y_umap'] = umap_2022[:, 1]

fig, axes = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('UMAP', fontsize = 10)
axes = axes.flatten()
for i, hue in enumerate(hues):
    ax = axes[i]
    sns.scatterplot(ax=ax, x='x_umap', y='y_umap', data=df_2022_norm, hue=hue, palette='Set2',
                    linewidth = 0.2)
    ax.set_title(f"{hue}", fontsize = 10, color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc = 'lower right', facecolor = 'white', edgecolor = 'none', fontsize = 7)
    ax.set_xlabel("t-SNE Component 1", size = 8, linespacing = 0.5)
    ax.set_ylabel("t-SNE Component 2", size = 8, linespacing = 0.5)
plt.show()



################################################## Cercle de corrélation PCA
fig, axes = plt.subplots(1, 2, figsize = (10,5), constrained_layout=True)
fig.suptitle('Corrélation des variables\n', fontsize = 15)

ax = axes[0]
ax.grid('whitegrid', color ='gray', alpha = 0.3)
ax.set_frame_on(False)
ax.tick_params(bottom=False, left=False)
ax.set_title("Cercle des corrélations", color ='dimgray')
for i in range (26):
    ax.arrow(0, 0, pca.components_[0,i], pca.components_[1,i], color = 'steelblue', lw = 0.5)
    ax.text(pca.components_[0,i]+0.08,pca.components_[1,i], df_2022_norm.columns[i+6], 
            size = 4, ha='center', va='bottom', color = 'dimgray')
ax.axhline(0, color='lightslategrey', lw = 0.5, linestyle = '--')
ax.axvline(0, color='lightslategrey', lw = 0.5, linestyle = '--')
circle = plt.Circle((0, 0), 1, color='lightslategrey', fill=False, lw = 0.7)
ax.add_artist(circle)  
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel(f"Principal Component 1 : {round(pca.explained_variance_ratio_[0],2)}%", size = 8, linespacing = 0.5)
ax.set_ylabel(f"Principal Component 2 : {round(pca.explained_variance_ratio_[1],2)}%", size = 8, linespacing = 0.5)

ax = axes[1]
ax.grid('whitegrid', color ='gray', alpha = 0.3)
ax.set_frame_on(False)
ax.tick_params(bottom=False, left=False)
ax.set_title("Zoom sur les variables", fontsize = 10, color ='dimgray')
for i in range (26):
    ax.arrow(0, 0, pca.components_[0,i], pca.components_[1,i], color = 'steelblue', alpha = 0.7)
    ax.text(pca.components_[0,i]+0.04,pca.components_[1,i], df_2022_norm.columns[i+6], 
            size = 5, ha='center', va='bottom', color = 'dimgray')
ax.axhline(0, color='lightslategrey', lw = 0.5, linestyle = '--')
ax.axvline(0, color='lightslategrey', lw = 0.5, linestyle = '--')
ax.set_xlabel(f"Principal Component 1 : {round(pca.explained_variance_ratio_[0],2)}%", size = 8, linespacing = 0.5)
ax.set_ylabel(f"Principal Component 2 : {round(pca.explained_variance_ratio_[1],2)}%", size = 8, linespacing = 0.5)

plt.show();



################################################## KMeans
fig, axes = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('K-means Clustering', fontsize = 12)
axes = axes.flatten()
for cluster in range(4):
    kmeans = KMeans(n_clusters = cluster+2, random_state = 0)
    clusters = kmeans.fit_predict(df_2022_norm_fit)
    centroids = kmeans.cluster_centers_
    ax = axes[cluster]
    sns.scatterplot(ax=ax, x='x_pca', y='y_pca', data=df_2022_norm, hue=clusters, palette='Set3',
                    edgecolor = 'none', alpha = 0.8)
    sns.scatterplot(ax=ax, x=centroids[:, 0], y=centroids[:, 1], color = 'darkred', marker = 'D')
    ax.set_title(f"{cluster+2} clusters", fontsize = 10, color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc = 'lower right', facecolor = 'white', edgecolor = 'none')
    ax.set_xlabel("PCA Component 1", size = 8, linespacing = 0.5)
    ax.set_ylabel("PCA Component 2", size = 8, linespacing = 0.5)
plt.show()



################################################## Méthode de coude
n_clusters = [1, 2, 3, 4, 5]  
distorsions = []

fig = plt.figure(figsize = (4,3))
ax = fig.add_subplot(111)
for cluster in n_clusters:
    clusters = KMeans(n_clusters = cluster, random_state = 0)
    clusters.fit(df_2022_norm_fit)
    distorsions.append(sum(np.min(cdist(df_2022_norm_fit, clusters.cluster_centers_, 'euclidean'), axis=1)) / np.size(df_2022_norm_fit, axis = 0))
plt.grid('whitegrid', color ='gray', alpha = 0.3)
plt.box(False)
plt.tick_params(bottom=False, left=False)
plt.plot(n_clusters, distorsions, 'x-', color = 'steelblue')
plt.xlabel('Nombre de Clusters', fontsize = 10)
plt.ylabel('Distorsion', fontsize = 10)
plt.title("Méthode du coude", color ='dimgray', fontsize = 10)
plt.xlim(0.5,5.5)
plt.tight_layout()
plt.show()



################################################## KMeans à 2 cluster
kmeans = KMeans(n_clusters = 2, random_state = 0)
clusters = kmeans.fit_predict(df_2022_norm_fit)
df_2022['cluster_kmeans'] = kmeans.labels_
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize = (4,3))
ax = fig.add_subplot(111)
sns.scatterplot(x='x_pca', y='y_pca', data=df_2022_norm, hue=clusters, palette='Set3',edgecolor = 'none', alpha = 0.8)
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color = 'darkred', marker = 'D')
ax.set_title(f"K-means à 2 clusters", fontsize = 10, color='dimgray')
ax.grid(color='gray', alpha=0.3)
ax.set_frame_on(False)
ax.tick_params(bottom=False, left=False)
ax.legend(loc = 'lower right', facecolor = 'white', edgecolor = 'none')
ax.set_xlabel("PCA Component 1", size = 8, linespacing = 0.5)
ax.set_ylabel("PCA Component 2", size = 8, linespacing = 0.5)
plt.tight_layout()
plt.show()



################################################## Comportement de vote
vote_by_cluster = df_2022[['voteG', 'voteCG', 'voteC', 'voteCD', 'voteD',
                           'cluster_kmeans']].groupby('cluster_kmeans').sum()
for column in ['voteG', 'voteCG', 'voteC', 'voteCD', 'voteD']:
    vote_by_cluster[f'p{column}'] = vote_by_cluster[column]/vote_by_cluster.sum(axis=1)
vote_by_cluster = vote_by_cluster[['pvoteG', 'pvoteCG', 'pvoteC', 'pvoteCD', 'pvoteD']]

fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 
clusters = vote_by_cluster.index
for i, cluster in enumerate(vote_by_cluster.index):
    data = vote_by_cluster.loc[cluster]
    axes[i].pie(data, labels=data.index, autopct='%1.1f%%', 
                wedgeprops={'edgecolor': 'w', 'linewidth': 0.5},
                colors = sns.color_palette("Set3", n_colors=5))
    axes[i].set_title(f'Cluster {cluster}', color = 'dimgrey', fontsize = 10)
plt.suptitle('Comportement de vote')
plt.tight_layout()
plt.show()



################################################## Eductaion
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
sns.boxplot(data=df_2022, x='pbac2022', y = 'cluster_kmeans', 
            orient = 'h', fill = False, palette = 'Set2')
plt.title("\n% Obtention de Bac\n", color ='dimgray', size = 10)
plt.grid('whitegrid', color ='gray', alpha = 0.3)
plt.box(False)
plt.tick_params(bottom=False, left=False)
ax.set_ylabel('Cluster', size = 8, linespacing = 0.5)
ax.set_xlabel('% Bac', size = 8, linespacing = 0.5) 
plt.legend(facecolor = 'white', edgecolor = 'none')
plt.tight_layout()
plt.show()



################################################## CSP
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
sns.countplot(x = 'cluster_kmeans', hue = 'csp', data = df_2022, palette = 'Set3')
plt.title("\nCatégorie socio-professionnelle de clusters\n", color ='dimgray', size = 10)
plt.grid('whitegrid', color ='gray', alpha = 0.3)
plt.box(False)
plt.tick_params(bottom=False, left=False)
ax.set_ylabel('Catégorie socio-professionnelle', size = 8, linespacing = 0.5)
ax.set_xlabel('Cluster', size = 8, linespacing = 0.5) 
plt.legend(loc = 'upper left', facecolor = 'white', edgecolor = 'none', fontsize = 7, bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()



################################################## Revenu
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
sns.boxplot(data=df_2022, x='revmoyadu2022', y = 'cluster_kmeans', 
            orient = 'h', fill = False, palette = 'Set2')
plt.title("\nRevenu\n", color ='dimgray', size = 10)
plt.grid('whitegrid', color ='gray', alpha = 0.3)
plt.box(False)
plt.tick_params(bottom=False, left=False, labelsize = 8)
ax.set_ylabel('Cluster', size = 8, linespacing = 0.5)
ax.set_xlabel('Revenu moyen', size = 8, linespacing = 0.5) 
plt.legend(facecolor = 'white', edgecolor = 'none')
plt.tight_layout()
plt.show()



################################################## PCA & KMeans au cours des années
dfs = [df_1981, df_1988, df_1995, df_2002, df_2007, df_2012, df_2017, df_2022]
years = [1981, 1995, 2002, 2007, 2012, 2017, 2022]
images = []

for df_year, year in zip(dfs, years):
    # Standardize all relevant columns
    all_cols = [col for col in df_year.columns if col not in ['dep', 'codecommune', 'vote_majoritaire', 
                                                              'participation', 'revenu', 'diplome', 
                                                              'proprietaire', 'csp', 'population']]
    scaler = StandardScaler()
    df_year_norm_fit = scaler.fit_transform(df_year[all_cols])
    # Merge the standardized columns back into the DataFrame
    df_year_norm = pd.concat([df_year[['dep', 'codecommune', 'vote_majoritaire', 'participation',
                                       'revenu', 'diplome', 'proprietaire', 'csp', 'population']],
                              pd.DataFrame(df_year_norm_fit, columns=all_cols)], axis=1)
    # PCA
    pca = PCA(n_components=2 )
    pca_year = pca.fit_transform(df_year_norm.drop(['dep', 'codecommune', 'vote_majoritaire', 
                                                    'participation', 'revenu', 'diplome', 
                                                    'proprietaire', 'csp', 'population'], axis=1))
    df_year_norm['x_pca'] = pca_year[:, 0]
    df_year_norm['y_pca'] = pca_year[:, 1]
    # K-means Clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(df_year_norm_fit)
    df_year['cluster_kmeans'] = kmeans.labels_
    centroids = pca.transform(kmeans.cluster_centers_)
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    fig.suptitle(f'{year}')
    axes = axes.flatten()

    ax = axes[2]
    sns.scatterplot(ax=ax, x='x_pca', y='y_pca', data=df_year_norm, hue=clusters, palette='Set3',
                    edgecolor='none', alpha=0.8)
    sns.scatterplot(ax=ax, x=centroids[:, 0], y=centroids[:, 1], color='darkred', marker='D')
    ax.set_title('K-Means Clusters', color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc='lower right', facecolor='white', edgecolor='none')
    ax.set_xlabel("Principal Component 1", size=8, linespacing=0.5)
    ax.set_ylabel("Principal Component 2", size=8, linespacing=0.5)

    ax = axes[1]
    sns.scatterplot(ax=ax, x='x_pca', y='y_pca', data=df_year_norm, hue='revenu', palette='Set3', 
                    linewidth=0.2)
    ax.set_title('Revenu', color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc='lower right', facecolor='white', edgecolor='none')
    ax.set_xlabel("Principal Component 1", size=8, linespacing=0.5)
    ax.set_ylabel("Principal Component 2", size=8, linespacing=0.5)
    
    ax = axes[0]
    sns.scatterplot(ax=ax, x='x_pca', y='y_pca', data=df_year_norm, hue='vote_majoritaire', palette='Set3',
                    edgecolor='none', alpha=0.8)
    ax.set_title('Vote', color='dimgray')
    ax.grid(color='gray', alpha=0.3)
    ax.set_frame_on(False)
    ax.tick_params(bottom=False, left=False)
    ax.legend(loc='lower right', facecolor='white', edgecolor='none')
    ax.set_xlabel("Principal Component 1", size=8, linespacing=0.5)
    ax.set_ylabel("Principal Component 2", size=8, linespacing=0.5)

# Enregistrement des images
    plt.savefig(f'{year}.png')
    images.append(f'{year}.png')
    plt.close(fig)



################################################## Visualisation

# List of image filenames with corresponding years
images = [
    ('1981.png', '1981'), 
    ('1988.png', '1988'), 
    ('1995.png', '1995'), 
    ('2002.png', '2002'), 
    ('2007.png', '2007'), 
    ('2012.png', '2012'), 
    ('2017.png', '2017'), 
    ('2022.png', '2022')
]

# Display each image with a title
for img, year in images:
    display(Image(filename=img))



################################################## Création de données revenu moyen total par année
col_revenu = []
for year in list(range(1980,2023)): 
    col_revenu.append('revmoyadu'+str(year))
revenu_rupture = revcommunes[col_revenu].mean(axis=0)
revenu_rupture.index = [year for year in range(1980,2023)]

plt.figure(figsize = (10,3))
plt.plot(revenu_rupture)
plt.title('Evolution de revenu en France\n')
plt.box(False)
plt.grid('whitegrid', alpha = 0.3)
plt.tick_params(bottom=False, left=False)
plt.ylabel('Revenu Moyen', size = 8, linespacing = 0.5)
plt.xlabel('Année', size = 8, linespacing = 0.5)
plt.tight_layout()
plt.show()



################################################## Détection de rupture avec une fonction de coût linéaire
rupture = rpt.Pelt(model = 'linear').fit(revenu_rupture.values.reshape(-1, 1)) # Modélisation
result_rupture = rupture.predict(pen = 1)
result_rupture = result_rupture[:-1]

model = LinearRegression().fit(revenu_rupture.index.values.reshape(-1,1), revenu_rupture.values.reshape(-1,1))
slope = model.coef_[0][0]
intercept = model.intercept_[0]
predicted_values = slope * revenu_rupture.index + intercept


rpt.show.display(revenu_rupture.values, result_rupture)
fig = plt.gcf()
ax = plt.gca()
fig.set_size_inches(10, 3.5)
#ax = plt.gca()
ax.plot(predicted_values, color="dimgray", alpha = 0.5, linestyle = '--')
ax.set_title('PELT avec une fonction de coût linéaire\n')
ax.set_frame_on(False)
ax.tick_params(bottom=False, left=False)
ax.set_xticks(range(43))
ax.set_xticklabels(revenu_rupture.index, rotation=90)
ax.set_ylabel('Revenu Moyen', size = 8, linespacing = 0.5)
ax.set_xlabel('Année', size = 8, linespacing = 0.5)
ax.grid('whitegrid', alpha = 0.3)
plt.tight_layout()
plt.show()



################################################## Détection de rupture avec une fonction de coût rbf
rupture = rpt.Pelt(model = 'rbf').fit(revenu_rupture.values) # Modélisation
result_rupture = rupture.predict(pen = 1)

rpt.show.display(revenu_rupture.values, result_rupture)
fig = plt.gcf()
ax = plt.gca()
fig.set_size_inches(10, 3.5)
ax.set_title('PELT avec une fonction de coût RBF \n')
ax.set_frame_on(False)
ax.tick_params(bottom=False, left=False)
ax.set_xticks(range(43))
ax.set_xticklabels(revenu_rupture.index, rotation=90)
ax.set_ylabel('Revenu Moyen', size = 8, linespacing = 0.5)
ax.set_xlabel('Année', size = 8, linespacing = 0.5)
plt.tight_layout()
plt.show()



