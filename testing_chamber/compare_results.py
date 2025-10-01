# %%
import os
import re
import pandas as pd

df = pd.read_csv("results_comparison_20250917_064315.csv")
df = df.dropna()
df = df.drop(columns=["similarity_improvement", "average_improvement"])

# %%

def find_locale_for_merge(locale: str) -> list:
    with open(os.path.join("merged_models", f"similarity_merge_{locale}", "merge_details.txt")) as f:
        text = f.read()

    local_pattern = re.compile(r"^\s*-\s*Locale: \s*([a-zA-Z]{2}-[a-zA-Z]{2})$", re.MULTILINE)
    found_locales = local_pattern.findall(text)

    return found_locales

# %%
df_nxn = pd.read_csv("nxn_results/nxn_eval_20250915_101911/evaluation_matrix.csv", index_col=0)
df_nxn = df_nxn.dropna()
df_nxn

# %%
scores_available = []
examples_locales_for_merge = find_locale_for_merge("vi-VN")
for locale in examples_locales_for_merge:
    if locale in df_nxn.index:
        scores_available.append(locale)
print(scores_available)

# %%
locales = "ka-GE"
scores_available = []
examples_locales_for_merge = find_locale_for_merge(locales)
for locale in examples_locales_for_merge:
    if locale in df_nxn.index:
        scores_available.append(locale)
benchmark = pd.concat([df_nxn[locales][scores_available], df[df["locale"] == locales][["baseline", "similarity", "average"]].squeeze(0)])
print(benchmark)


