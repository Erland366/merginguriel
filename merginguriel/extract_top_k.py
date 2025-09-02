import pandas as pd
from merginguriel.utils import get_all_positive_columns

def load_df():
    df = pd.read_csv("./big_assets/language_similarity_matrix.csv", index_col=0)
    return df

def main():
    df = load_df()
    print(f"{df.head() = }")
    """
    df.head() =           mif       lex       sbc       tzb       bli       ayc  ...       cwa       adr       gbp       ewo       jac       ttr
mif  1.000000  0.735980  0.827329  0.552823  0.741467  0.439388  ...  0.720577  0.752618  0.857215  0.773953  0.564103  0.923077
lex  0.735980  1.000000  0.883883  0.552176  0.666973  0.394120  ...  0.648181  0.953959  0.659570  0.677644  0.594445  0.707673
sbc  0.827329  0.883883  1.000000  0.547997  0.657411  0.342997  ...  0.638889  0.870388  0.757033  0.694444  0.613825  0.773953
tzb  0.552823  0.552176  0.547997  1.000000  0.479301  0.676661  ...  0.493197  0.572364  0.613388  0.575396  0.763422  0.579148
bli  0.741467  0.666973  0.657411  0.479301  1.000000  0.441176  ...  0.914659  0.686644  0.612056  0.886076  0.521773  0.686544

[5 rows x 4005 columns]"""
    # top_k_df = get_all_positive_columns("eng", df)
    # print(f"{top_k_df = }")
    # """top_k_df = ['kri', 'pcm', 'enm', 'ang', 'ydd']"""

if __name__ == "__main__":
    main()