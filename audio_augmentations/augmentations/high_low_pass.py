import random
try:
    import essentia.standard
except Exception as e:
    print("Essentia not found")


class HighLowPass:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        highlowband = random.randint(0, 1)
        if random.random() < self.p:
            if highlowband == 0:
                highpass_freq = random.randint(200, 1200)
                filt = essentia.standard.HighPass(
                    cutoffFrequency=highpass_freq, sampleRate=self.sr
                )
            elif highlowband == 1:
                lowpass_freq = random.randint(2200, 4000)
                filt = essentia.standard.LowPass(
                    cutoffFrequency=lowpass_freq, sampleRate=self.sr
                )
            # else:
            #     filt = essentia.standard.BandPass(bandwidth=1000, cutoffFrequency=1500, sampleRate=self.sr)
            audio = filt(audio)
            

        return audio
