import pandas
from sklearn.model_selection import train_test_split

from Veri import reader


class OnIsleme():
    data = reader.Reader().getData()

    def veri_isleme(self):
        # Verinin başlıklarını yazıyorum.
        self.data.columns = [
                'id',
                'Clump Thickness',
                'Uniformity of Cell Size',
                'Uniformity of Cell Shape',
                'Marginal Adhesion',
                'Single Epithelial Cell Size',
                'Bare Nuclei',
                'Bland Chromatin',
                'Normal Nucleoli',
                'Mitoses',
                'Class'
            ]
        # Boş verilerin bulunduğu satırları kaldırıyorum.
        self.data.dropna(inplace=True, axis = 0)
        print(self.data.values[0])
        # Eğitimde Id kolonuna ihtiyacım olmayacak.
        self.data.drop(columns=['id'], inplace=True)

        # Sınıf isimlerini yönelgeye göre değiştiriyorum.
        self.data['Class'] = \
            self.data['Class'].replace({2 : 'benign', 4 : 'malignant'})

    def seti_bol(self):
        self.veri_isleme()
        X = self.data.drop(columns=['Class']).values
        Y = self.data['Class'].values
        return train_test_split(X,Y, random_state=42)