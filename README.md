# HSE_Credit_scoring

Задача кредитного скоринга в рамках будкемпа ВШЭ.
Датасет содержит информацию о клиентах некоторого банка. Ваша задача состоит в том, чтобы по различным характеристикам клиентов спрогнозировать целевую переменную - имел клиент просрочку 90 и более дней или нет (и если имел, то банк не будет выдавать кредит этому клиенту, а иначе будет).

# Целевая переменная #

*SeriousDlqin2yrs*: клиент имел просрочку 90 и более дней

# Признаки #

- RevolvingUtilizationOfUnsecuredLines: общий баланс средств (total balance on credit cards and personal lines of credit except real estate and no installment debt
like car loans divided by the sum of credit limits)

* age: возраст заемщика
* NumberOfTime30-59DaysPastDueNotWorse: сколько раз за последние 2 года наблюдалась просрочка 30-59 дней
* DebtRatio: ежемесячные расходы (платеж по долгам, алиментам, расходы на проживания) деленные на месячный доход
* MonthlyIncome: ежемесячный доход
* NumberOfOpenCreditLinesAndLoans: количество открытых кредитов (напрмер, автокредит или ипотека) и кредитных карт
* NumberOfTimes90DaysLate: сколько раз наблюдалась просрочка (90 и более дней)
* RealEstateLoansOrLines: закодированное количество кредиов (в том числе под залог жилья) - чем больше код буквы, тем больше кредитов
* NumberOfTime60-89DaysPastDueNotWorse: сколько раз за последние 2 года заемщик задержал платеж на 60-89 дней
* NumberOfDependents: количество иждивенцев на попечении (супруги, дети и др)
* GroupAge: закодированная возрастная группа - чем больше код, тем больше возраст

## Создание web-приложения ##

https://hsecreditscoring-pf35zae8zrj.streamlit.app/
