import pandas as pd

class Reader():
    data = None

    def getData(self):
        if self.data == None:
            self.data = pd.read_csv(
                'Veri/data.csv'
            )
        return self.data