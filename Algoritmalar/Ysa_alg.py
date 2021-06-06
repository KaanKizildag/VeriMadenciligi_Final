from sklearn.neural_network import MLPClassifier
import Veri.veri_isleme


class YsaAlg():
    kararAlgoritmasi = MLPClassifier(activation='relu')

    def __init__(self):
        isleme = Veri.veri_isleme.OnIsleme()
        x_train, x_test, y_train, y_test = isleme.seti_bol()
        self.kararAlgoritmasi.fit(x_train, y_train)

    def tahmin(self, x):
        return self.kararAlgoritmasi.predict(x)
