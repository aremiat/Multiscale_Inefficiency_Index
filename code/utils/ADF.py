from statsmodels.tsa.stattools import adfuller

# Dickey-Fuller Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] < 0.05:
        print("The series is stationary")
        return result[1]
    else:
        print("The series is not stationary, differencing recommended")
        return result[1]
