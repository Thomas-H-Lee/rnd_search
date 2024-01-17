#%%
import numpy as np
import pandas as pd
import seaborn as sns

from bokeh.plotting import figure, show

rs_file = r'D:\Workspace\5. 기술기획\17. DT\02. 기술관리시스템 DB활용성\기술활용실적_v1_220630.xlsx'  

df = pd.read_excel(rs_file, index_col=False)  

#%%
df_groupby_id = df.groupby(['id', 'year'], as_index=False)['benefit'].agg('count')
df_pivot = df_groupby_id.pivot('year', 'id', 'benefit')
df_pivot.plot(kind ='barh')

#%%
sns.pairplot(df_groupby_id, y_vars='tech_name', x_vars=['year'], diag_kind=None)


#%%
df_groupby_total = df.groupby(['id', 'year'])
df_groupby_total.plot.bar()
# df.boxplot(figsize=(10,4))

#%%
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = np.array([ [r, g, 150] for r, g in zip(50 + 2*x, 30 + 2*y) ], dtype="uint8")

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

p = figure(tools=TOOLS)

p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=0.6,
          line_color=None)

show(p)

# %%
