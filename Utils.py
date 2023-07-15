#%%
import researchpy
from scipy.stats import shapiro, mannwhitneyu, chi2, normaltest

def shapiro_w(df):
    stat, p = shapiro(df)
    print('Estatística de Teste: {:.4f}, valor p: {}'.format(stat, p))
    if p > 0.05:
        return print('Não há evidência suficiente para rejeitar a hipótese de normalidade.')
    else:
        return print('A hipótese de normalidade é rejeitada.')
#%%
def mannwhitney_u(df1, df2, alternative):
    stat, p = mannwhitneyu(df1, df2, alternative=alternative)
    print("Estatística de teste U: ", stat)
    print("Valor p: ", p)
    alpha = 0.05

    if p < alpha:
        return print("Diferença estatisticamente significante")
    else:
        return print("Não há diferença estatisticamente significante")
#%%
def chi_2(df):
    prob = 0.95
    critical = chi2.ppf(prob, len(df) - 1)
    alpha = 1.0 - prob
    stat, p = normaltest(df)
    print("Critical: ", critical)
    print("Estatística de teste U: ", stat)
    print("Valor p: ", p)

    if p < alpha:
        return print("Diferença estatisticamente significante")
    else:
        return print("Não há diferença estatisticamente significante")
#%%
def research_py(df1, df2, test):
    result = researchpy.crosstab(df1, df2, test=test)
    return print(result)
#%%