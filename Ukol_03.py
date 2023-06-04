"""
Stáhni si data ze souboru Life-Expectancy-Data-Updated.csv, která udávají průměrnou dobu života v jednotlivých zemích světa. Data pocházejí od Světové zdravotnické organizace (WHO) a Světové banky. Vytvoř regresní model, jehož úkolem bude zjistit, které faktory ovlivňují průměrnou délku života.

Vyber data pro jeden konkrétní rok (např. pro rok 2015).
Vysvětlovanou proměnnou ve tvém modelu bude Life expectancy, což je průměrná délka života.
"""

import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

#stažení dat ze souboru
data = pd.read_csv('Life-Expectancy-Data-Updated.csv')

#vytvoření dataFrame s daty pro rok 2015
data2015 = data[data['Year'] == 2015]

#zobrazení náhledu dat vytvořeného dataFrame pro rok 2015
print(data2015.head())

#test dat sloupců Life expectancy a GDP_per_capita, zda se jedná o data v normálním rozdělení. Test provedeme Shapiro-Wilkovým testem a stanovíme si tyto hypotézy:

#Nulová hypotéza: Hodnoty mají normální rozdělení.
#Alternativní hypotéza: Hodnoty nemají normální rozložení.

res_swt_lifeexp = st.shapiro(data2015['Life_expectancy'])
print(f"Toto je výsledek Shapiro-Wilkova testu pro data Life expectancy: {res_swt_lifeexp}")

res_swt_gdppercap = st.shapiro(data2015['GDP_per_capita'])
print(f"Toto je výsledek Shapiro-Wilkova testu pro data GDP_per_capita: {res_swt_lifeexp}")

#Toto je výsledek Shapiro-Wilkova testu pro data Life_expectancy: ShapiroResult(statistic=0.9528260231018066, pvalue=1.0978403224726208e-05)
#Toto je výsledek Shapiro-Wilkova testu pro data GDP_per_capita: ShapiroResult(statistic=0.9528260231018066, pvalue=1.0978403224726208e-05)
#Závěr testu: Na základě výsledku Shapiro-Wilkova testu, kde p-value je větší než hladina významnosti 0,05, nulovou hodnotu nezamítáme a můžeme konstatovat, že nulová hypotéza je u obou sloupců dat platná a data mají normální rozdělení. 

#Graf
g = sns.regplot(data2015, x='GDP_per_capita', y='Life_expectancy', scatter_kws={"s": 1}, line_kws={"color":"r"})
plt.show()


#zobrazemí tabulky s využitím metody využitím modulu scipy a metody summary() a zjisti koeficient determinace.
#Nulová hypotéza: GDP per capita nemá vliv na Life expectancy.
#Alternativní hypotéza: GDP per capita má vliv na Life expectancy.

formula = "Life_expectancy ~ GDP_per_capita"
mod = smf.ols(formula=formula, data=data2015)
res = mod.fit()
print(res.summary())

"""
Výsledky:
 R-squared:                       0.396
===============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         67.9538      0.561    121.219      0.000      66.848      69.060
GDP_per_capita     0.0003   2.58e-05     10.774      0.000       0.000       0.000
"""
# Při hladině významnosti a regresním koeficientu 0,0003 můžeme tvrdit, že existuje statisticky významný vliv mezi GDP per capita a Life expectancy. P-value je nižší než hladina významnosti 0,05, proto tedy nulovou hypotézu zamítáme. Koeficient determinace má hodnotu 0,396, tzn. že 39,6 % variability průměrné délky života (Life expectancy) je vysvětleno vysvětlující proměnnou (GDP per capita)v našem regresním modelu. 

"""
Do modelu přidej následující sloupce:

Schooling - průměrná délka studia (v letech),
Incidents_HIV - nákazy virem HIV (počet případů na osobu)
Diphtheria - procento populace očkované proti záškrtu,
Polio - procento populace očkované proti dětské obrně,
BMI - průměrný BMI index populace,
Measles - procento populace očkované proti spalničkám.
U každého sloupce se zamysli nad tím, jestli může délku života výrazně ovlivnit a jaké tipuješ znaménko koeficientu (kladné - zvyšuje délku života, záporné - snižuje délku života).

Sestav model z vybraných sloupců a proveď následující kroky:
"""

#Nulová hypotéza: Žádná z přidáných proměnných (GDP_per_capita, Schooling, Incidents_HIV, Diphtheria, Polio, BMI, Measles) nemá vliv na průměrnou délku života (Life_expectancy).
#Alternativní hypotéza: Alespoň jedna z přidáních proměnných má vliv na průměrnou délku života.

formula = "Life_expectancy ~ GDP_per_capita + Schooling + Incidents_HIV + Diphtheria + Polio + BMI + Measles"
mod2 = smf.ols(formula=formula, data=data2015)
res2 = mod2.fit()
print(res2.summary())

"""
OLS Regression Results
==============================================================================
Dep. Variable:        Life_expectancy   R-squared:                       0.790
Model:                            OLS   Adj. R-squared:                  0.782
Method:                 Least Squares   F-statistic:                     92.03
Date:                Sun, 04 Jun 2023   Prob (F-statistic):           1.22e-54
Time:                        22:56:13   Log-Likelihood:                -482.13
No. Observations:                 179   AIC:                             980.3
Df Residuals:                     171   BIC:                             1006.
Df Model:                           7
Covariance Type:            nonrobust
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         37.9443      4.016      9.449      0.000      30.018      45.871
GDP_per_capita     0.0001   1.96e-05      5.565      0.000    7.05e-05       0.000
Schooling          0.8445      0.146      5.791      0.000       0.557       1.132
Incidents_HIV     -1.4128      0.173     -8.154      0.000      -1.755      -1.071
Diphtheria        -0.0035      0.051     -0.067      0.946      -0.105       0.098
Polio              0.1385      0.060      2.304      0.022       0.020       0.257
BMI                0.4254      0.161      2.646      0.009       0.108       0.743
Measles            0.0390      0.023      1.731      0.085      -0.005       0.083
==============================================================================
Omnibus:                        3.894   Durbin-Watson:                   1.994
Prob(Omnibus):                  0.143   Jarque-Bera (JB):                3.965
Skew:                          -0.347   Prob(JB):                        0.138
Kurtosis:                       2.774   Cond. No.                     3.19e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.19e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

#Interpretace výsledků:
#Ve výsledcích regresní analýzy vidíme, že proměnná "Diphtheria" nemá statisticky významný vliv na délku života ("Life_expectancy"). P-value pro tuto proměnnou je 0.946, což je vyšší než hladina významnosti 0.05. Na základě tohoto výsledku nemůžeme zamítnout nulovou hypotézu, která říká, že proměnná "Diphtheria" nemá vliv na délku života. Ostatní proměnné mají p-value nižší než hladina významnosti 0,05 a tedy nulovou hypotézu můžeme zamítnout a tvrdit, že dané proměnné vliv na Life expectancy vliv mají.
# Záporné regresní koeficienty u počtu případů nakažených HIV a procento proočkované populace proti záškrtu by měly ukazovat, že mají vliv na snižování věku dožití. Procento proočkované populace proti záškrtu dle p-value nemá ale na délku života vliv. Ostatní proměnné mají na délku života pozitivní vliv.
#Koeficient determinace 0,790 vyjadřuje, že 79 % variability věku dožití (závislé proměnné) je vysvětleno vysvětlujícími proměnnými (GDP_per_capita, Schooling, Incidents_HIV, Diphtheria, Polio, BMI, Measles). Zbývajících 21 % variability může být ovlivněno jinými faktory, které nejsou zahrnuty do modelu. Koeficient determinace pouze pro vysvětlující proměnnou GDP per capita byl 0,39, tedy s přidáním proměnných se zvýšil a snížila se variabilita, jejíž specifikace nebyla známa a také je vidět, že GDP per capita vysvětluje podstatnou část variability závislé proměnné. 
#P-value reziduí (0.143 a 0.138) jsou hodnoty vyšší než hladina významnosti 0.05, takže nemůžeme zamítnout nulovou hypotézu o normalitě reziduí. To naznačuje, že rezidua modelu mají přibližně normální rozdělení.

"""
Pokud jsi nezamítl(a) hypotézu normality, podívej se do sloupce P>|t| a vyber řádek s nejvyšší p-hodnotou. Koeficient pro daný řádek odeber z modelu. Jak se změnila hodnota ostatních koeficientů? Jak se změnil koeficient determinace?
"""

formula3 = "Life_expectancy ~ GDP_per_capita + Schooling + Incidents_HIV + Polio + BMI + Measles"
mod3 = smf.ols(formula=formula3, data=data2015)
res3 = mod3.fit()
print(res3.summary())

"""
Výsledek:
OLS Regression Results
==============================================================================
Dep. Variable:        Life_expectancy   R-squared:                       0.790
Model:                            OLS   Adj. R-squared:                  0.783
Method:                 Least Squares   F-statistic:                     108.0
Date:                Sun, 04 Jun 2023   Prob (F-statistic):           1.12e-55
Time:                        23:55:13   Log-Likelihood:                -482.14
No. Observations:                 179   AIC:                             978.3
Df Residuals:                     172   BIC:                             1001.
Df Model:                           6
Covariance Type:            nonrobust
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         37.9505      4.003      9.481      0.000      30.049      45.852
GDP_per_capita     0.0001   1.95e-05      5.594      0.000    7.06e-05       0.000
Schooling          0.8455      0.145      5.844      0.000       0.560       1.131
Incidents_HIV     -1.4129      0.173     -8.179      0.000      -1.754      -1.072
Polio              0.1349      0.026      5.106      0.000       0.083       0.187
BMI                0.4260      0.160      2.661      0.009       0.110       0.742
Measles            0.0389      0.022      1.737      0.084      -0.005       0.083
==============================================================================
Omnibus:                        3.921   Durbin-Watson:                   1.993
Prob(Omnibus):                  0.141   Jarque-Bera (JB):                3.993
Skew:                          -0.348   Prob(JB):                        0.136
Kurtosis:                       2.773   Cond. No.                     3.19e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.19e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Po odstranění nezávislé proměnné Diptheria nedošlo ke změně koeficientu determinace, stejně jako k výraznějším změnám u koeficientů ostatních nezávislých proměnných. Tato skutečnost potvrzuje předešlé zjištění, že nezávislá proměnná Diptheria nemá na Life expectance významný statistický vliv.