import os
import random
from .base import BaseDataset
import librosa
from pathlib import Path
import numpy as np

class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.inference = opt.inference
        self.eval_image = opt.eval_image
        

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]
        # print(infos)
        # break
        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        instrument = infos[0][0].split('/')[-2]
        prev_instruments = [instrument]

        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            # make sure no overlap target instruments
            instrument = self.list_sample[indexN][0].split('/')[-2]
            while instrument in prev_instruments:
                indexN = random.randint(0, len(self.list_sample)-1)
                instrument = self.list_sample[indexN][0].split('/')[-2]
            infos[n] = self.list_sample[indexN]
            prev_instruments.append(instrument)
        
        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)

        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{}.jpg'.format(center_frameN + idx_offset)))
                        #'{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

        
        try:
            for n, infoN in enumerate(infos):
                # use image dataset for eval
                if self.eval_image == 1:
                    instrument = path_frames[n][0].split('/')[-3]
                    use_wrong_image = False
                    use_blank_image = False
                    if use_wrong_image:
                        instrument_list = ['acoustic_guitar', 'cello', 'clarinet', 'trumpet', 'violin', 'flute', 'accordion']
                        instrument_list.remove(instrument)
                        inst_idx = random.randint(0, len(instrument_list)-1)
                        instrument = instrument_list[inst_idx]

                    if use_blank_image:
                        path_image = ['../../../../dataset/instrument_image/' + 'blank.jpg'] * len(path_frames[n])
                    elif self.inference == 1:
                        path_image = [path_frames[0][0][:-8]]* len(path_frames[n])
                    else:
                        path_image = ['../../../../dataset/instrument_image/' + instrument + '/' + instrument + '_0.jpg'] * len(path_frames[n])
                    frames[n] = self._load_frames(path_image)

                else:
                    frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            
            
       
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)


        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)
        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
        return ret_dict
