from sklearn import tree

import Veri.veri_isleme


class DecisonTreeAlg():

    kararAlgoritmasi = tree.DecisionTreeClassifier()
    isleme = Veri.veri_isleme.OnIsleme()
    x_train, x_test, y_train, y_test = isleme.seti_bol()

    kararAlgoritmasi.fit(x_train, y_train)
    
    def tahmin(self, x):
        return self.kararAlgoritmasi.predict(x)