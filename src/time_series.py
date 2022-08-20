class TimeSeries:
    def __init__(self, name, data, frequency, seasonality):
        self.name = name
        self.data = data
        self.frequency = frequency
        self.seasonality = seasonality

    def __str__(self):
        return self.name + ' (Frequency: ' + str(self.frequency) + ', Seasonality: ' + str(
            self.seasonality)+ ')'

    def __repr__(self):
        return self.__str__()
