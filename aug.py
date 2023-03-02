# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchaudio
import augment
import argparse

from dataclasses import dataclass

class RandomPitchShift:
    def __init__(self, shift_max=300):
        self.shift_max = shift_max

    def __call__(self):
        return np.random.randint(-self.shift_max, self.shift_max)

class RandomClipFactor:
    def __init__(self, factor_min=0.0, factor_max=1.0):
        self.factor_min = factor_min
        self.factor_max = factor_max
    def __call__(self):
        return np.random.triangular(self.factor_min, self.factor_max, self.factor_max)

@dataclass
class RandomReverb:
     reverberance_min: int = 50
     reverberance_max: int = 50
     damping_min: int = 50
     damping_max: int = 50
     room_scale_min: int = 0
     room_scale_max: int = 100

     def __call__(self):
          reverberance = np.random.randint(self.reverberance_min, self.reverberance_max + 1)
          damping = np.random.randint(self.damping_min, self.damping_max + 1)
          room_scale = np.random.randint(self.room_scale_min, self.room_scale_max + 1)

          return [reverberance, damping, room_scale]

class SpecAugmentBand:
     def __init__(self, sampling_rate, scaler):
          self.sampling_rate = sampling_rate
          self.scaler = scaler

     @staticmethod
     def freq2mel(f):
          return 2595. * np.log10(1 + f / 700)

     @staticmethod
     def mel2freq(m):
          return ((10.**(m / 2595.) - 1) * 700)

     def __call__(self):
          F = 27.0 * self.scaler
          melfmax = freq2mel(self.sample_rate / 2)
          meldf = np.random.uniform(0, melfmax * F / 256.)
          melf0 = np.random.uniform(0, melfmax - meldf)
          low = mel2freq(melf0)
          high = mel2freq(melf0 + meldf)
          return f'{high}-{low}'


def augmentation_factory(description, sampling_rate):

     t_ms=50
     pitch_shift_max=300
     pitch_quick=True
     room_scale_min=0
     room_scale_max=100
     reverberance_min=50
     reverberance_max=50
     damping_min=50
     damping_max=50
     clip_min=0.5
     clip_max=1.0

     chain = augment.EffectChain()
     description = description.split(',')

     for effect in description:
          if effect == 'bandreject':
               chain = chain.sinc('-a', '120', SpecAugmentBand(sampling_rate, band_scaler))
          elif effect == 'pitch':
               pitch_randomizer = RandomPitchShift(pitch_shift_max)
               if pitch_quick:
                    chain = chain.pitch('-q', pitch_randomizer).rate('-q', sampling_rate)
               else:
                    chain = chain.pitch(pitch_randomizer).rate(sampling_rate)
          elif effect == 'reverb':
               randomized_params = RandomReverb(reverberance_min, reverberance_max, 
                                   damping_min, damping_max, room_scale_min, room_scale_max)
               chain = chain.reverb(randomized_params).channels()
          elif effect == 'time_drop':
               chain = chain.time_dropout(max_seconds=t_ms / 1000.0)
          elif effect == 'clip':
               chain = chain.clip(RandomClipFactor(clip_min, clip_max))
          elif effect == 'none':
               pass
          else:
               raise RuntimeError(f'Unknown augmentation type {effect}')
     return chain

def get_augment_wav(x):
     
     sampling_rate = 16000
     chain = "pitch,clip,reverb"
     augmentation_chain = augmentation_factory(chain, sampling_rate)

     y = augmentation_chain.apply(x, 
               src_info=dict(rate=sampling_rate, length=x.size(1), channels=x.size(0)),
               target_info=dict(rate=sampling_rate, length=0)
     )
     aug_audio = y.detach().numpy()[0]
     # print("============")
     # print(x.size())
     # print(aug_audio.shape)

     return aug_audio
     # print(y.detach().numpy().shape)

if __name__ == '__main__':
     x, _ = torchaudio.load("Dataset/normalized_dataset_2feb/test_labels/crowd_scream-00166.wav")
     print(get_augment_wav(x).shape)