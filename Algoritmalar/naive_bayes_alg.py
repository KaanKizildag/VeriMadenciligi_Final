from sklearn.naive_bayes import GaussianNB

from Veri.veri_isleme import OnIsleme


class NaiveBayes():
    kararAlgoritmasi = GaussianNB()

    def __init__(self):
        isleme = OnIsleme()
        x_train, x_test, y_train, y_test = isleme.seti_bol()
        self.kararAlgoritmasi.fit(x_train, y_train)

    def tahmin(self, x):
        return self.kararAlgoritmasi.predict(x)
