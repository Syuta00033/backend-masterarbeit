from scipy.signal import butter, filtfilt


class KalmanFilter1D:

    def __init__(self, process_variance=0.001, measurement_variance=0.01):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        self._initialized = False

    def update(self, measurement):
        if not self._initialized:
            self.posteri_estimate = measurement
            self._initialized = True
            return measurement

        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        return self.posteri_estimate


class _SmoothedLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class LandmarkSmoother:

    NUM_LANDMARKS = 33

    def __init__(self, process_variance=0.001, measurement_variance=0.01):
        self.filters_x = [KalmanFilter1D(process_variance, measurement_variance) for _ in range(self.NUM_LANDMARKS)]
        self.filters_y = [KalmanFilter1D(process_variance, measurement_variance) for _ in range(self.NUM_LANDMARKS)]
        self.filters_z = [KalmanFilter1D(process_variance, measurement_variance) for _ in range(self.NUM_LANDMARKS)]

    def smooth(self, landmarks):
        result = []
        for i, lm in enumerate(landmarks):
            x = self.filters_x[i].update(lm.x)
            y = self.filters_y[i].update(lm.y)
            z = self.filters_z[i].update(lm.z)
            result.append(_SmoothedLandmark(x, y, z))
        return result


def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)
