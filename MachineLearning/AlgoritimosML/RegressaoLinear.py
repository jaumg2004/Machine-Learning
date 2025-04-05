import numpy as np

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__correlation_coefficient = self.__correlacao()
        self.__inclination = self.__inclinacao()
        self.__intercept = self.__interceptacao()

    def __correlacao(self):
        covariacao = np.cov(self.x, self.y, bias=True)[0][1]
        variancia_x = np.var(self.x)
        variancia_y = np.var(self.y)

        return covariacao/np.sqrt(variancia_x*variancia_y)

    def __inclinacao(self):
        stdx = np.std(self.x)
        stdy = np.std(self.y)

        return self.__correlation_coefficient * (stdy/stdx)

    def __interceptacao(self):
        mediax = np.mean(self.x)
        mediay = np.mean(self.y)

        return  mediay - mediax*self.__inclination

    def previsao(self, valor):

        return self.__intercept + (self.__inclination * valor)


x = np.array([1,2,3,4,5])
y = np.array([2,4,5,8,10])

lr = LinearRegression(x,y)
previsao = lr.previsao(13)
print(previsao)