class TimeSeries:
    def __init__(self, name, data, seasonality, frequency):
        self.name = name
        self.data = data
        self.frequency = frequency
        self.seasonality = seasonality
