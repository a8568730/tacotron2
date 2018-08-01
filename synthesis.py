# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pylab as plt
# import IPython.display as ipd

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from os import path

# %matplotlib inline

# def plot_data(data, figsize=(16, 4)):
#     fig, axes = plt.subplots(1, len(data), figsize=figsize)
#     for i in range(len(data)):
#         axes[i].imshow(data[i], aspect='auto', origin='bottom', 
#                        interpolation='none')

# Setup hparams
hparams = create_hparams("distributed_run=False,mask_padding=False")
hparams.sampling_rate = 22050
hparams.filter_length = 1024
hparams.hop_length = 256
hparams.win_length = 1024


# Load model from checkpoint
checkpoint_path = path.join(dirname(__file__), "cp_outdir", "checkpoint_40000")
model = load_model(hparams)
try:
    model = model.module
except:
    pass
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path)['state_dict'].items()})
_ = model.eval()


# Prepare text input
text = "This is an example of text to speech synthesis after 14 hours training."
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

# Decode text input and plot results
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)


# Load TacotronSTFT and convert mel-spectrogram to spectrogram
taco_stft = TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length, 
    sampling_rate=hparams.sampling_rate)
mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
spec_from_mel_scaling = 1000
spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
spec_from_mel = spec_from_mel * spec_from_mel_scaling

# Synthesize audio from spectrogram using the Griffin-Lim algorithm
waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), 
                       taco_stft.stft_fn, 60)
# ipd.Audio(waveform[0].data.cpu().numpy(), rate=hparams.sampling_rate)
print('waveform=', waveform[0].data.cpu().numpy())

