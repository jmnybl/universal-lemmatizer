import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from collections import OrderedDict
from matplotlib.ticker import AutoMinorLocator

with open("udv2.2_treebank_info.json", "rt") as f: # json with various information calculated from the UD v2.2 treebanks
    info=json.load(f)

languages=OrderedDict()
for key in sorted(info.keys()):
    _,lang=key.split("_",1) # extract plain language name from key
    lang,_=lang.rsplit("-",1)
    treebank_code=info[key]["treebank_code"]

    if "running tokens (form) ambiguous on train" not in info[key]: # no ambiguous data for this treebank
        print("No data for",key,"skipping...")
        continue
    if lang not in languages:
        languages[lang]=([],[],[])
    languages[lang][0].append(treebank_code)
    languages[lang][1].append(info[key]["running tokens (form) ambiguous on train"])
    languages[lang][2].append(info[key]["running tokens (form+tag) ambiguous on train"])

for key in languages.keys():
    print(key,":",languages[key])


# average scores for each language (do not show individual treebanks as there is too many)
average_scores=OrderedDict()
for lang in languages.keys():
    form=sum([s for s in languages[lang][1]])/len(languages[lang][1])
    tag=sum([s for s in languages[lang][2]])/len(languages[lang][2])
    average_scores[lang]={}
    average_scores[lang]["ambiguous token-tag pairs"]=tag
    average_scores[lang]["ambiguous tokens"]=form

# turn dictionary into pandas dataframe
df=pd.DataFrame(average_scores)
df=df.reindex(["ambiguous tokens", "ambiguous token-tag pairs"]) # fix row order

df['Average'] = df.mean(numeric_only=True, axis=1) # add Average column

print(df)
df=df.T # transpose table
df.plot.bar(figsize=(10,5))#,colors=['g','orange']) # make bar plot
ax = plt.axes()
ax.minorticks_on() # show minor vertical lines
ax.set_axisbelow(True) # set grid behind bars
minor_locator = AutoMinorLocator(2) # set n-1 minor vertical lines between every major line
ax.yaxis.set_minor_locator(minor_locator)
ax.yaxis.grid(True, which="major", color="lightgrey", linestyle="-", linewidth=0.5, alpha=0.8) # set style for major vertical lines
ax.yaxis.grid(True, which="minor", color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.4) # set style for minor vertical lines

plt.tight_layout() # fit x-axis labels to the screen
plt.savefig('ambiguous_words.png', bbox_inches='tight') # save with tight borders
plt.show() # finally show the plot


