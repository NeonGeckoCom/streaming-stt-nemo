import numpy as np
import resampy
import torch

from nemo.collections.asr.models import EncDecCTCModelBPE



class Model:
    hf_model = "stt_en_citrinet_512_gamma_0_25"


    def __init__(self):
        self.stt_model = EncDecCTCModelBPE. \
                    from_pretrained(self.hf_model, map_location="cpu")


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