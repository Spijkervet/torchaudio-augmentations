import random
import torch

try:
    import essentia.standard
except Exception as e:
    print("Essentia not found")


class HighLowPass(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, audio):
        highlowband = random.randint(0, 1)
        try:
            if highlowband == 0:
                highpass_freq = random.randint(200, 1200)
                filt = essentia.standard.HighPass(
                    cutoffFrequency=highpass_freq, sampleRate=self.sample_rate
                )
            elif highlowband == 1:
                lowpass_freq = random.randint(2200, 4000)
                filt = essentia.standard.LowPass(
                    cutoffFrequency=lowpass_freq, sampleRate=self.sample_rate
                )            
            # else:
            #     filt = essentia.standard.BandPass(bandwidth=1000, cutoffFrequency=1500, sampleRate=self.sample_rate)
            
            audio = audio.numpy().reshape(-1)
            audio = filt(audio)
            audio = torch.from_numpy(audio).reshape(1, -1)
        except:
            print("Essentia library not found, skipping HighLowPass augmentation")
        return audio
