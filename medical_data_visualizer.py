import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# 4
df = df[(df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))]

# 5
def draw_cat_plot():
    # 6
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 7
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    df_cat['value'] = df_cat['value'].astype(int)  # Convert 'value' column to integer

    # 8
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    g.set_axis_labels("variable", "total")

    # 9
    fig = plt.gcf()

    # 10
    fig.savefig('catplot.png')
    return fig

# 11
def draw_heat_map():
    # 12
    df_heat = df.copy()

    # 13
    corr = df_heat.corr()

    # 14
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 15
    fig, ax = plt.subplots(figsize=(12, 10))

    # 16
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # 17
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # 18
    fig.savefig('heatmap.png')
    return fig

if __name__ == "__main__":
    draw_cat_plot()
    draw_heat_map()
