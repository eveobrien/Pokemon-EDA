import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tests = pd.read_csv('tests.csv')
combats = pd.read_csv('combats.csv')
df = pd.read_csv('/Users/eveobrien/Desktop/practice_eda/pokemon.csv')

df.info()

df.corr()

#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

df.columns

#Tips for me to remember
#Line plot is better when x axis is time.
#Scatter is better when there is correlation between two variables
#Histogram is better when we need to see distribution of numerical data.
#Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = style of line
df.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()

# Scatter Plot
# x = attack, y = defense
df.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot
plt.show()

#Correlation shows that Pokemon seem to have similar attack statistics to defense statistics
#But there are lots of wild anomolies that need to be cleaned  later

# Histogram
# bins = number of bar in figure
df.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()

# clf() = cleans it up again you can start a fresh
df.Speed.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()

#Tip for me to remember - Dictionary is faster than using lists

series = df['Defense']        # data['Defense'] = series
print(type(series))
data_frame = df[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))

# 1 - Filtering Pandas data frame
x = df['Defense']>200     # There are only 3 pokemon who have higher defense value than 200
df[x]

df[np.logical_and(df['Defense']>200, df['Attack']>100 )]  # There are only 2 pokemon who have higher defence value than 200 and higher attack value than 100

# This is also same with previous code line. Therefore we can also use '&' for filtering.
#So, Mega Steelix and Mega Aggron have the highest Defence  and Attack  Powers.
df[(df['Defense']>200) & (df['Attack']>100)]


#Using the average, classify pokemon with their Speed, creating a 'speed_level' feature for high and low
threshold = sum(df.Speed)/len(df.Speed)
df["speed_level"] = ["high" if i > threshold else "low" for i in df.Speed]
df.loc[:10,["speed_level","Speed"]]

#This is where the fun begins...
#Let's explore the Pokemon with the highest stats (Most Powerful!)

df['Total_stats'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
print(df.iloc[:, [1, -1]].head())

#visualise
sns.distplot(df.Total_stats)
plt.show()

#Whats the average stats a pokemon has?

mean_stats = df['Total_stats'].mean()
print(mean_stats)

#These are the 'most average' pokemon

#closest value from near_stats
average_pokemon = min(df['Total_stats'], key=lambda x: abs(x-mean_stats))
print(df.loc[((df['Total_stats'] >= average_pokemon-5) & (df['Total_stats'] <= average_pokemon+5)), ['Name', 'Total_stats']])

#Now, lets explore our top 10 weakest Pokemon
sorted_pokemon_df = df.sort_values(by='Total_stats')
print(sorted_pokemon_df[['Name', 'Total_stats']].head(10))
#Poor Sunkern....

#Now, our top 10 strongest pokemon!

print(sorted_pokemon_df[['Name', 'Total_stats', 'Legendary']].tail(10))
#Two Mega Mewtwo's and a Rayquaza are the victors.

#Now, we have 6 generations of Pokemon in our data. I wonder if Pokemon have been made stronger as time goes on

#group data by generation
group_df = df.drop(['#', 'Legendary'], axis=1)
pokemon_groups = group_df.groupby('Generation')
pokemon_groups_mean = pokemon_groups.mean()

sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['Total_stats'])
plt.show()
#Gen 4 (my personal favourite) seem to have the highest stats out of our Pokemon data.


# C L E A N I N G  T I M E
# so many anomalies....

df.info()

#Let's explore the pokemon types and their frequency

print(df['Type 1'].value_counts(dropna =False))

#Most pokemon are Water or Normal types, with the least common type being Flying by far!

#Using Describe we can see a ton about that data, including averages, standard deviation, ranges etc!

df.describe()

#Tips  for me to remember!
#Box plots: visualize basic statistics like outliers, min/max or quantiles

df.boxplot(column='Attack', by = 'Legendary')


#To create a thinner, longer dataframe  thats easier to read at the moment, we use Melting, and  to undo it, pivoting!
#Im not actually using this 'Melting' method at the moment I just wanted practice to look back on

# id_vars = what we do not wish to melt
# value_vars = what we want to melt

#melted = pd.melt(frame=df_new,id_vars = 'Name', value_vars= ['Attack','Defense'])


# MISSING DATA!
#Eventually I'm onto dealing with missing data. I can drop them, fill them in with fillna(), or fill with averages,

#df.info() #Name has 1 missing  value, Type 2 has !386! missing values!

df.loc[df['Name'].isnull()==True] #Its ID is 63, and with some research I found the missing name was 'Primeape'. So lets add that

df['Type 2'] = df['Type 2'].fillna('None')
df['Name'] = df['Name'].fillna('Primeape')

#There seems to be a mistake in the types too, there are some types named 'Fight' instead of 'Fighting',
#which is the correct terminology for Pokemon
df['Type 1'] = df['Type 1'].replace('Fighting', 'Fight')
df['Type 2'] = df['Type 2'].replace('Fighting', 'Fight')

#Let's change T/F to 1/0 for the Legendary pokemon too
df['Legendary'] = df['Legendary'].map({False: 0, True:1})


#VISUAL EDA

# Plotting all data
df.loc[:,["Attack","Defense","Speed"]]
df.plot()

#Confusing! Sub plotting will be better
df.plot(subplots = True)
plt.show()
#Defense varies a ton

#Scatter Plot, lets look at correlation
df.plot(kind = "scatter", x = "Attack", y="Defense")
plt.show()

#Let's visualse the total pokemon of each type, including legendary pokemon
#This will show us the most common types in our dataset, and what types legendary pokemon are most common / least common in
#Using Typ1 1 as it is the pokemons primary type
sns.set_color_codes("pastel")
ax = sns.countplot(x="Type 1", hue="Legendary", data=df)
plt.xticks(rotation=90)
plt.xlabel('Type 1')
plt.ylabel('Total ')
plt.title("Total Pokemon by Type 1")
plt.show()
#This visulisation shows Psychic and Dragon types are most likely to include legendary pokemon in our dataset!


#STATISTICAL EDA

#Can describe the data again with:
df.describe()

#Let's now do some investigation into the Win Rate of each Pokemon when in a battle through the Combats Dataset
total_wins = combats.Winner.value_counts()
#no of wins for each pokemon
numberOfWins = combats.groupby('Winner').count()

countByFirst = combats.groupby('Second_pokemon').count()
countBySecond = combats.groupby('First_pokemon').count()
print("count by first winner shape: " + str(countByFirst.shape))
print("count by second winner shape: " + str(countBySecond.shape))
print("Total wins shape: "  + str(total_wins.shape))
#no of dimensions is different so there must be a pokemon that didnt win a single fight
#lets find it!

#.setdiff1d() finds the set difference of two arrays
#.iloc[] integer-location based indexing for selection by position.

detect_losing_pokemon = np.setdiff1d(countByFirst.index.values, numberOfWins.index.values)-1 #offset as index and number are off by 1
losing_pokemon = df.iloc[detect_losing_pokemon[0],]
print(losing_pokemon)
#Shuckle has been found as the pokemon who did not win a single battle

#EDA CONCLUSIONS

# 1) Correlation shows that Pokemon seem to have similar attack statistics to defense statistics
# 2) There are only 3 pokemon who have higher defense value than 200 & There are only 2 pokemon who have higher defence value than 200 and higher attack value than 100
    # Those being Mega Steelix & Mega Aggron
# 3) Gen 4 seem to have the highest stats out of our Pokemon data.
    # Gen 1 was the weakest, Gen 4 was the peak, then a rapid decline to a another increase in Gen 6
# 4) Most pokemon are Water or Normal types, with the least common type being Flying by far!
# 5) Psychic and Dragon types are most likely to include legendary pokemon







