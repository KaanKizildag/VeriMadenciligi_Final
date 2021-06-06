from sklearn.metrics import confusion_matrix

class SoruCevaplayan():
    gercekDegerler = []
    tahminler = []
    fn = 0
    fp = 0
    tp = 0
    tn = 0

    def __init__(self, gercekDegerler:list, tahminler: list):
        self.gercekDegerler = gercekDegerler
        self.tahminler = tahminler
        self.tn, self.fp, self.fn, self.tp = self.confusionMatris().ravel()


    def confusionMatris(self):
        return confusion_matrix(self.gercekDegerler, self.tahminler)

    def get_tn(self):
        return self.tn

    def get_tp(self):
        return self.tp

    def get_fn(self):
        return self.fn

    def get_fp(self):
        return self.fp

    def get_dogruluk(self):
        return (self.get_tp() + self.get_tn()) / (self.get_tp() + self.get_tn() + self.get_fp() + self.get_fn())

    def get_recall(self):
        '''recall TP / (TP + FN)'''
        return self.get_tp() / (self.get_tp() + self.get_fn())
    def get_precision(self):
        '''precision TP / (TP + FP)'''
        return self.get_tp() / (self.get_tp() + self.get_fp())