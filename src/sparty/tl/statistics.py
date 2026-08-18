from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    wilcoxon,
)
from statsmodels.stats.multitest import multipletests


def check_normality(values, alpha=0.05) -> dict:
    values = values.dropna()

    if len(values) < 3:
        return {
            "normal": False,
            "pvalue": None
        }
    
    stat, p = shapiro(values)

    return {
        "statistic": stat,
        "pvalue": p,
        "normal": p > alpha
    }


def check_variance(groups, alpha=0.05) -> dict:

    stat, p = levene(
        *groups
    )

    return {
        "statistic":stat,
        "pvalue":p,
        "equal_variance": p > alpha
    }

    
def ttest(grp1,grp2):
    stat, p = ttest_ind(
                grp1,
                grp2,
                equal_var=False
            )
    
    return {
        group_col: group,
        "group1": comb[0],
        "group2": comb[1],
        "method": "t-test",
        "statistic": stat,
        "pvalue": p,
        "n_group1": len(grp1),
        "n_group2": len(grp2),
        "mean_group1": grp1.mean(),
        "mean_group2": grp2.mean(),
    }

def mannwhitney(grp1,grp2):
    stat, p = mannwhitneyu(
        grp1,
        grp2,
        alternative="two-sided"
    )
        
    return {
        group_col: group,
        "group1": comb[0],
        "group2": comb[1],
        "method": "Mann-Whitney",
        "statistic": stat,
        "pvalue": p,
        "n_group1": len(grp1),
        "n_group2": len(grp2),
        "mean_group1": grp1.mean(),
        "mean_group2": grp2.mean(),
    }


    
# def anova(
        
# )
    
# def kruskal(
        
# )

# def glm_binomial()
# def beta_binomial()

# def tukey_test()
# def dunn_test()

# def multiple_testing()