import numpy as np
import resampy
import torch

from nemo.collections.asr.models import EncDecCTCModelBPE

from .configs import languages


class Model:
    langs = languages


    def __init__(self, lang="en"):
        self.stt_model = EncDecCTCModelBPE. \
                    from_pretrained(self.langs[lang]["model"], map_location="cpu")
        self.freeze_model()
        

    def freeze_model(self):
        self.stt_model.preprocessor.featurizer.dither = 0.0
        self.stt_model.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        self.stt_model.eval()
        # Freeze the encoder and decoder modules
        self.stt_model.encoder.freeze()
        self.stt_model.decoder.freeze()

    def stt(self, audio_buffer: np.array, sr: int):
        audio_fp32 = self._to_float32(audio_buffer)
        audio_16k = self._resample(audio_fp32, sr)

        logits, logits_len, greedy_predictions = self.stt_model.forward(
            input_signal=torch.tensor([audio_16k]), 
            input_signal_length=torch.tensor([len(audio_16k)])
        )

        current_hypotheses, all_hyp = self.stt_model.decoding.ctc_decoder_predictions_tensor(
            logits, decoder_lengths=logits_len, return_hypotheses=False,
        )
        return current_hypotheses


    def _resample(self, audio_fp32: np.array, sr: int):
        audio_16k = resampy.resample(audio_fp32, sr, self.stt_model.cfg.sample_rate)
        return audio_16k


    def _to_float32(self, audio_buffer: np.array):
        audio_fp32 = np.divide(audio_buffer, np.iinfo(audio_buffer.dtype).max, dtype=np.float32)
        return audio_fp32